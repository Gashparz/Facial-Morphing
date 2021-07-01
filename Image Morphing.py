import glob
import os
import pickle
import stat
from operator import itemgetter
from random import randrange
import shutil
import cv2 as cv
import numpy as np
import dlib
from hair_segmentation import hair_segmentation
from age_detection import age_detect

age_model = pickle.load(open('age-model-final.pkl', 'rb'))


def get_points(img, flag):
    points = landmarks_detections(img)
    points = list(points)
    points.append([0, 0])
    points.append([img.shape[1] - 1, 0])
    points.append([0, img.shape[0] - 1])
    points.append([img.shape[1] - 1, img.shape[0] - 1])
    points.append([0, int((img.shape[0] - 1) / 2)])
    points.append([img.shape[1] - 1, int((img.shape[0] - 1) / 2)])
    points.append([int((img.shape[1] - 1) / 2), 0])
    points.append([int((img.shape[1] - 1) / 2), img.shape[0] - 1])

    extTop, extLeft, extLeftM, extRight, extRightM = hair_segmentation.getContour(img)

    if flag == "yes":
        print("Using hairsegmentation")
        points.append(extTop)
        points.append(extLeft)
        points.append(extLeftM)
        points.append(extRight)
        points.append(extRightM)

        return points
    elif flag == "no":
        print("Not using hairsegmentation")
        return points


def landmarks_detections(img):
    img_gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    detections = detector(img_gray)

    points = []

    for detection in detections:
        # x1 = detection.left()
        # y1 = detection.top()
        # x2 = detection.right()
        # y2 = detection.bottom()
        # cv.rectangle(img, (x1,y1), (x2, y2), (0, 255, 255), 2)
        landmarks = predictor(image=img_gray, box=detection)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # cv.circle(img, (x,y), 4, (0, 0, 255), -1)
            points.append([x, y])

    points = np.array(points)
    # cv.imshow("cal", img)
    # cv.waitKey(0)
    return points


def crop_triangle(img, tri):
    tri_crop = []

    tri_rect = np.float32([[tri[0][0], tri[0][1]], [tri[1][0], tri[1][1]], [tri[2][0], tri[2][1]]])

    rect = cv.boundingRect(tri_rect)

    img_crop = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

    for i in range(0, 3):
        tri_crop.append([(tri_rect[i][0] - rect[0]), (tri_rect[i][1] - rect[1])])

    # cv.imshow("img_crop", img_crop)
    # cv.waitKey(0)
    # cv.imshow("img_crop", np.float32(tri_crop))
    # cv.waitKey(0)
    return img_crop, np.float32(tri_crop), rect


def warp_triangle(img, points1, img2, points2, img_avg, points_avg, delaunay_indices, t):
    for i1, i2, i3 in delaunay_indices:
        triangle1 = [points1[i1], points1[i2], points1[i3]]
        triangle2 = [points2[i1], points2[i2], points2[i3]]
        triangle_avg = [points_avg[i1], points_avg[i2], points_avg[i3]]

        img1_crop, tri1_crop, rect1 = crop_triangle(img, triangle1)
        img2_crop, tri2_crop, rect2 = crop_triangle(img2, triangle2)
        img_avg_crop, tri_crop_avg, rect_avg = crop_triangle(img_avg, triangle_avg)

        affine_transformation = cv.getAffineTransform(tri1_crop, tri_crop_avg)
        warped = cv.warpAffine(img1_crop, affine_transformation, (rect_avg[2], rect_avg[3]), None,
                               flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
        affine_transformation2 = cv.getAffineTransform(tri2_crop, tri_crop_avg)
        warped2 = cv.warpAffine(img2_crop, affine_transformation2, (rect_avg[2], rect_avg[3]), None,
                                flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

        mask = np.zeros((rect_avg[3], rect_avg[2], 3), dtype=np.float32)

        cv.fillConvexPoly(mask, np.int32(tri_crop_avg), (1.0, 1.0, 1.0), 16, 0)

        imgRect = (1.0 - t) * warped + t * warped2
        img_avg[rect_avg[1]:rect_avg[1] + rect_avg[3], rect_avg[0]:rect_avg[0] + rect_avg[2]] = \
            img_avg[rect_avg[1]:rect_avg[1] + rect_avg[3], rect_avg[0]:rect_avg[0] + rect_avg[2]] * (
                    1 - mask) + imgRect * mask

        # img_avg_crop *= 1 - mask
        # img_avg_crop += imgRect * mask


def delaunay(img, points):
    rectangle = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv.Subdiv2D(rectangle)

    for p in points:
        subdiv.insert((p[0], p[1]))

    triangleList = subdiv.getTriangleList()
    triangles = []
    # draw_delaunay(img, subdiv)
    # img_voronoi = np.zeros(img.shape, dtype=img.dtype)
    # draw_voronoi(img_voronoi, subdiv)
    # cv.imshow("cal", img_voronoi)
    # cv.waitKey(0)

    for tri in triangleList:
        vertexes = [0, 0, 0]
        for x in range(3):
            xx = x * 2
            for i in range(len(points)):
                # print(p[vv + 1], points[i][1])
                if tri[xx] == points[i][0] and tri[xx + 1] == points[i][1]:
                    vertexes[x] = i

        triangles.append(vertexes)

    return triangles, subdiv


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_voronoi(img, subdiv):
    (facets, centers) = subdiv.getVoronoiFacetList([])
    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)
            ifacet = np.array(ifacet_arr, np.int)
            color = (randrange(255), randrange(255), randrange(255))
            cv.fillConvexPoly(img, ifacet, color, cv.LINE_AA, 0)
            ifacets = np.array([ifacet])
            cv.polylines(img, ifacets, True, (0, 0, 0), 1, cv.LINE_AA, 0)
            cv.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv.FILLED, cv.LINE_AA, 0)


def draw_delaunay(img, subdiv):
    triangleList = subdiv.getTriangleList()
    r = (0, 0, img.shape[1], img.shape[0])

    for tri in triangleList:
        pt1 = (tri[0], tri[1])
        pt2 = (tri[2], tri[3])
        pt3 = (tri[4], tri[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv.line(img, pt1, pt2, (255, 255, 255), 1, cv.LINE_AA, 0)
            cv.line(img, pt2, pt3, (255, 255, 255), 1, cv.LINE_AA, 0)
            cv.line(img, pt3, pt1, (255, 255, 255), 1, cv.LINE_AA, 0)

    cv.imshow("cal", img)
    cv.waitKey(0)


def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    print("Would you like to use hair segmentation? yes or no")
    choice = input()

    frames_path = "C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/prezentare/imagini_intermediare/"
    output_path = "C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/prezentare/output/"
    shutil.rmtree(frames_path, onerror=remove_readonly)
    shutil.rmtree(output_path, onerror=remove_readonly)

    os.mkdir(frames_path)
    os.mkdir(output_path)

    images = []
    for name in glob.glob('C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/prezentare/input/*.jpg'):
        image, predicted_age = age_detect.age_detection(name, age_model)
        if image is None:
            images.append((None, 0))
        else:
            images.append((name, predicted_age))

    sorted_images = sorted(images, key=itemgetter(1))

    for sorted_image in sorted_images:
        print(str(sorted_image[0]) + " the predicted age was: " + str(sorted_image[1]))

    for i in range(len(sorted_images)):
        if sorted_images[i][0] is None:
            print("In the image number: " + str(i + 1) + " no face was detected!\nSkipping!")
            continue

        if i == len(sorted_images) - 1:
            break

        img1 = cv.imread(sorted_images[i][0])
        img2 = cv.imread(sorted_images[i + 1][0])

        h1, w1, c1 = img1.shape
        h2, w2, c2 = img2.shape
        if w1 < w2:
            img2 = cv.resize(img2, (w1, h1))
        else:
            img1 = cv.resize(img1, (w2, h2))

        points1 = get_points(img1, choice)
        points2 = get_points(img2, choice)

        nr_frames = 60
        alpha_values = np.linspace(0, 100, int(nr_frames))
        for (f, a) in enumerate(alpha_values):
            t = float(a) / 100

            points_avg = []
            for j in range(0, len(points1)):
                x = (1 - t) * points1[j][0] + t * points2[j][0]
                y = (1 - t) * points1[j][1] + t * points2[j][1]
                points_avg.append((x, y))
            points_avg = np.array(points_avg)
            tri, subdiv = delaunay(img1, points1)

            imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

            warp_triangle(img1, points1, img2, points2, imgMorph, points_avg, tri, t)

            index = str(i + 1) + (str(f + 1).zfill(4))

            cv.imwrite(frames_path + index + '.jpg', np.uint8(imgMorph))

    img_array = []
    for filename in glob.glob('C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/prezentare/imagini_intermediare/*.jpg'):
        img = cv.imread(filename)
        height, width, layers = img.shape
        v_size = (width, height)
        img_array.append(img)

    out = cv.VideoWriter('C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/prezentare/output/timelapse.avi', cv.VideoWriter_fourcc(*'DIVX'), 10, v_size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    main()
