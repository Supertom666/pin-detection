//code by tomLee, 2021/11/16
#include <opencv2/opencv.hpp>
#include <iostream>
// using namespaces to nullify use of cv::function(); syntax and std::function();

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

 

int center_x = 382;
int center_y = 108;
int width = 48;
int height = 110;


// float y_up_direction_filter = 0.4;
float y_up_direction_filter = 0.08;
float y_down_direction_filter = 0.9;
float x_right_direction_filter = 0.9;
float x_left_direction_filter = 0.1;

int x_offset = 20; // 20 pixels offset from the right width of the image
int filter_area_size =3000; // 20000 pixels filter size for the bounding box
int filter_area_size_wt =10; /// 10  for the weighting the bounding box
int time_of_dilate = 0; // time of dilate minus the time of erode
//light enhancement
float light_enhance_alpha = 1.1;
int light_enhance_beta = 10;
bool show_filter_image = false; // show the filter box
Rect default_box(center_x, center_y, width, height); // (x=100, y=100)为框的左上角，10x10的大小

//identity light enhancement
float isLight(cv::Mat& src) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    float sum = 0;
    int ls[256] = {0};
    int size = gray.rows * gray.cols;

    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            uchar pixelValue = gray.at<uchar>(i, j);
            sum += (pixelValue - 128);
            ls[(int)pixelValue]++;
        }
    }

    float avg = sum / size;
    float total = 0;

    for (int i = 0; i < 256; i++) {
        total += abs((float)i - avg) * ls[i];
    }
    float mean = total / size;

    float cast = abs(avg / mean);
    cout << "Average:" << avg << ", Anomaly value:" << cast << endl;

    if (cast > 1) {
        cout << "Brightness Anomaly: " << (avg > 0 ? "Too Bright" : "Too Dark") << " " << avg << endl;
    } else {
        cout << "Normal" << endl;
    }

    return avg;
}

//enhance light
void Brightnessand_contrast(Mat &image , Mat& outputImage, double alpha, int beta)
{
	//调整亮度和对比度
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			for (int c = 0; c < image.channels(); c++) {
				outputImage.at<cv::Vec3b>(y, x)[c] =
					cv::saturate_cast<uchar>(alpha * image.at<cv::Vec3b>(y, x)[c] + beta);
			}
		}
	}
}



// 判断一个点是否在指定矩形框内
bool isPointInsideRect(const Point& pt, const Rect& box) {
    return (pt.x >= box.x && pt.x <= box.x + box.width && 
            pt.y >= box.y && pt.y <= box.y + box.height);
}

void drawBoundingBoxes(const Mat& binaryImage, Mat& outputImage, const Rect& init_box) {
    // Ensure the image is binary (0 or 255 values)
    Point bottomLeft_pt, bottomRight_pt;
    Point upright_pt, downright_pt;
    // Point bottomRight;
    Mat binImage;
    threshold(binaryImage, binImage, 100, 255, THRESH_BINARY);


    // 计算 init_box 的中心点
    Point initCenter(init_box.x + init_box.width / 2, init_box.y + init_box.height / 2);
    

    // Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Variable to store the bounding box with maximum y value
    Rect maxXYBoundingBox;
    int maxY = -1;  // Start with a very small y value to find the max
    int maxheight = -1;
    int maxx = -1;  // Start with a very small y value to find the max
    int maxwidth = -1;

    // Iterate through each contour and find the bounding box with maximum y
    for (const auto& contour : contours) {
        Rect boundingBox = boundingRect(contour);
        // cout << "Bounding boundingBox: " << boundingBox.area() << endl;
        // Skip bounding boxes with area smaller than 100
        if (boundingBox.area() < filter_area_size || abs((boundingBox.y+(boundingBox.height/2)) - (init_box.y+(init_box.height/2))) > init_box.height || boundingBox.area() > filter_area_size_wt*filter_area_size) {
            if (show_filter_image ==true) {
                rectangle(outputImage, boundingBox, Scalar(255, 255, 0), 2);
            };
            // rectangle(outputImage, boundingBox, Scalar(255, 255, 0), 2);
            rectangle(outputImage, boundingBox, Scalar(0, 0, 0), FILLED);
            continue;
        }
        
        cout << "Bounding boundingBox: " << boundingBox.area() << endl;
        // Check if this bounding box's y value is the largest encountered
        if (boundingBox.y+boundingBox.height > maxY+maxheight&& boundingBox.x+boundingBox.width > maxx+maxwidth) 
        {
            maxY = boundingBox.y;
            maxheight = boundingBox.height;
            maxx = boundingBox.x;
            maxwidth = boundingBox.width;
            maxXYBoundingBox = boundingBox;
        }

        // Draw the bottom edge of the bounding box
        // Point bottomLeft(maxXYBoundingBox.x, maxXYBoundingBox.y + maxXYBoundingBox.height); // Bottom-left corner
        // Point bottomRight(maxXYBoundingBox.x + maxXYBoundingBox.width, maxXYBoundingBox.y + maxXYBoundingBox.height); // Bottom-right corner
        Point upright_pt(maxXYBoundingBox.x+maxXYBoundingBox.width, maxXYBoundingBox.y ); // Bottom-left corner
        Point downright_pt(maxXYBoundingBox.x + maxXYBoundingBox.width, maxXYBoundingBox.y + maxXYBoundingBox.height); // Bottom-right corner
        // line(outputImage, bottomLeft, bottomRight, Scalar(0, 0, 255), 2); // Red line for the bottom edge
        line(outputImage, upright_pt, downright_pt, Scalar(0, 0, 255), 2); // Red line for the bottom edge
        // bottomLeft_pt = bottomLeft;
        // bottomRight_pt = bottomRight;
        bottomLeft_pt = upright_pt;
        bottomRight_pt = downright_pt;

    }
    imshow("filter_area_size", outputImage);
    // 计算 bottomLeft_pt 和 bottomRight_pt 的中点
    Point middlePoint((bottomLeft_pt.x + bottomRight_pt.x) / 2, 
                      (bottomLeft_pt.y + bottomRight_pt.y) / 2);

    // cout<< "Bounding Boxes2: " << bottomLeft_pt << " " << bottomRight_pt << endl;


    rectangle(outputImage, init_box, Scalar(255, 255, 0), 2);
    
    // isPointInsideRect(bottomRight, box); // true
    // cout<< "isPointInsideRect: " << isPointInsideRect(bottomLeft_pt, init_box) << endl;

    // 输出 init_box 的中心点和 middlePoint
    cout << "Init Box Center: " << initCenter << endl;
    cout << "Middle Point: " << middlePoint << endl;

    // 计算距离
    int dis = sqrt(pow(middlePoint.x - initCenter.x, 2) + pow(middlePoint.y - initCenter.y, 2));
    int dis_y = abs(init_box.y - middlePoint.y);
    line(outputImage, middlePoint, initCenter, Scalar(255, 0, 0), 2);


    if (dis_y < init_box.width/2) {
        cout << "OK satifies the condition" << endl;
    }
    else {
        cout << "Not OK does not satisfy the condition" << endl;
    };
    cout << "Distance: " << dis << endl;

}   


int main()
{
    // Reading image
    // Mat orignal_img = imread("test.jpg");
    // Mat orignal_img = imread("test.bmp");
    // Mat orignal_img = imread("4.bmp");
    // Mat orignal_img = imread("test5.bmp");
    Mat orignal_img = imread("test7.bmp");
    if (orignal_img.empty()) {
        cerr << "Error: Could not open or find the image!" << endl;
        return -1;
    }
    Mat show_orignal_img = orignal_img; 
    // Mat orignal_img = imread("test6.bmp");
    cout << "Original Image Size: " << orignal_img.size() << endl;
    // float light_value;
    float light_value = isLight(orignal_img);
    

    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    for( int x = 0; x < orignal_img.rows; x++ ) {
      for( int y = 0; y < orignal_img.cols; y++ ) {
          if ( orignal_img.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
            orignal_img.at<Vec3b>(x, y)[0] = 0;
            orignal_img.at<Vec3b>(x, y)[1] = 0;
            orignal_img.at<Vec3b>(x, y)[2] = 0;
          }
        }
    }
    // // Show output image
    // imshow("Black Background Image", orignal_img);
    // Mat kernel = Mat_<float>(3,3);
    Mat Light_img;

    int cnt = 0;
    int time_of_brightness = 0;
    cout << "Light Value: " << light_value << endl;
    if(light_value<50){
        Light_img = orignal_img.clone();
        for(cnt = 0; cnt < 10; cnt++){
            
            Brightnessand_contrast(Light_img, Light_img, light_enhance_alpha, light_enhance_beta);
            float light_value2 = isLight(Light_img);
            cout << "Light light_value2: " << light_value2 << endl;
            if(light_value2>0){
                orignal_img = Light_img;
                time_of_brightness =cnt;
                break;
            }
        }

    }


    cout << "Dark light_value: " << light_value << endl;

    // Create a kernel that we will use for accuting/sharpening our image
    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -9, 1,
            1,  1, 1); // an approximation of second derivative, a quite strong kernel

    Mat imgLaplacian;
    Mat sharp = orignal_img; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);

    orignal_img.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow( "New Sharped Image", imgResult );
    orignal_img = imgResult; // copy back
    // Create binary image from source image
    Mat bw;
    cvtColor(orignal_img, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 200, CV_THRESH_BINARY | CV_THRESH_OTSU);

    cv::Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	cv::Mat result;
    cv::Mat result2;
	cv::erode(bw, result, element);
    // cv::erode(result, result, element);

    cv::dilate(result, result2, element);
    int ii = 0;
    if (time_of_brightness > 0) {
        for(ii=0 ; ii<time_of_brightness-time_of_dilate; ii++){
            // cout << "time_of_brightness: " << time_of_brightness << endl;
            cv::dilate(result2, result2, element);
        }
    }

    // Convert to grayscale
    Mat img_gray;
    cvtColor(orignal_img, img_gray, COLOR_BGR2GRAY);
    
    // Blur the image for better edge detection
    Mat img_blur;
    GaussianBlur(img_gray, img_blur, Size(3,3), 0);
    
    // Mat binaryImage1;
    // threshold(img_blur, binaryImage1, 100, 255, THRESH_BINARY); // Adjust the threshold value as needed
    // threshold(img_blur, binaryImage1, 100, 225, CV_THRESH_BINARY); // Adjust the threshold value as needed
    
    //using the result2 dilation as binaryImage1
    Mat binaryImage1 = result2;
    // imshow("binaryImage1", binaryImage1);
    for(int i=0; i<binaryImage1.rows; i++){


        for(int j=0; j<binaryImage1.cols; j++){           
            // y direction filter for removing horizontal lines
            if(i<(orignal_img.rows*y_up_direction_filter) || (i>(orignal_img.rows*y_down_direction_filter)))
            {

                binaryImage1.at<uchar>(i,j) = 0;
                      
                if( j>(orignal_img.cols-x_offset)){          
                    binaryImage1.at<uchar>(i,j) = 0;
                }

            }
            // x direction filter for removing horizontal lines
            if(j>(orignal_img.cols*x_right_direction_filter) || j<(orignal_img.cols*x_left_direction_filter)){          
                binaryImage1.at<uchar>(i,j) = 0;
            }
            
        }
    }
    // Draw bounding boxes around white regions
    Mat outputImage;
    outputImage = orignal_img.clone(); // Copy the original image to outputImage
    cvtColor(binaryImage1, outputImage, COLOR_GRAY2BGR); // Convert to BGR to draw colored bounding boxes
    
    drawBoundingBoxes(binaryImage1, outputImage ,default_box);

    // Display the output image with bounding boxes
    line(outputImage, Point(0,orignal_img.rows*y_up_direction_filter), Point(binaryImage1.cols, orignal_img.rows*y_up_direction_filter), Scalar(255,0,0), 1);
    line(outputImage, Point(0,orignal_img.rows*y_down_direction_filter), Point(binaryImage1.cols, orignal_img.rows*y_down_direction_filter), Scalar(255,0,0), 1);
    line(outputImage, Point(orignal_img.cols*x_left_direction_filter,0), Point(orignal_img.cols*x_left_direction_filter, binaryImage1.rows), Scalar(255,0,255), 1);
    line(outputImage, Point(orignal_img.cols*x_right_direction_filter,0), Point(orignal_img.cols*x_right_direction_filter, binaryImage1.rows), Scalar(255,0,255), 1);
    Mat imgConcat;
    hconcat(show_orignal_img, outputImage, imgConcat);

    // 显示拼接后的图像
    imshow("Two Images", imgConcat);

    waitKey(0);
    destroyAllWindows();
    return 0;
}