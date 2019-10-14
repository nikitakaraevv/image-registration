#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    //Load grayscale images
	Mat I1 = imread("../IMG_0045.JPG", IMREAD_GRAYSCALE);
	Mat I2 = imread("../IMG_0046.JPG", IMREAD_GRAYSCALE);

	imshow("I1", I1);
	imshow("I2", I2);

    //Detect keypoints and compute descriptors using AKAZE
	Ptr<AKAZE> D = AKAZE::create();
	
	vector<KeyPoint> kpts1, kpts2, matched1, matched2;
    Mat desc1, desc2;
    
    D->detectAndCompute(I1, noArray(), kpts1, desc1);
    D->detectAndCompute(I2, noArray(), kpts2, desc2);
	
	
	Mat J; //to save temporal images
    drawKeypoints(I1, kpts1, J);
    imshow("Keypoints 1", J);
    imwrite("Keypoints1.jpg", J);
    drawKeypoints(I2, kpts2, J);
    imshow("Keypoints 2", J);
    imwrite("Keypoints2.jpg", J);
    
    
    //Use brute-force matcher to find 2-nn matches
	BFMatcher M(NORM_HAMMING);
    vector< vector<DMatch> > knn_matches;
    vector<DMatch> good_matches;
    M.knnMatch(desc2, desc1, knn_matches, 2);
    const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
    
    
    //Use 2-nn matches to find correct keypoint matches
    for(size_t i = 0; i < knn_matches.size(); i++) {
        DMatch nearest = knn_matches[i][0];
        double dist1 = knn_matches[i][0].distance;
        double dist2 = knn_matches[i][1].distance;
        if(dist1 < dist2*nn_match_ratio) {
            int new_i = static_cast<int>(matched1.size());
            matched1.push_back(kpts1[nearest.trainIdx]);
            matched2.push_back(kpts2[nearest.queryIdx]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    
    drawMatches(I1, matched1, I2, matched2, good_matches, J);
	imshow("Matches", J);
    imwrite("Matches.jpg", J);
    
    //Use matches to compute the homography matrix
    std::vector<Point2f> first;
    std::vector<Point2f> second;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        first.push_back( matched1[i].pt);
        second.push_back( matched2[i].pt);
    }
    Mat H = findHomography(second, first, RANSAC, 5.0);
    
    
    //Apply the computed homography matrix to warp the second image
    Mat Panorama(I1.rows, 2 * I1.cols,  CV_8U);
    warpPerspective(I2, Panorama, H, Panorama.size());
    
    //Combining the result with the first image
    for(int i = 0; i < I1.rows; i++){
        for(int j = 0; j < I1.cols; j++)
            Panorama.at<uchar>(i,j) = I1.at<uchar>(i,j);
    }
    imshow("Panorama", Panorama);
    imwrite("Panorama.jpg", Panorama);

	waitKey(0);
	return 0;
}
