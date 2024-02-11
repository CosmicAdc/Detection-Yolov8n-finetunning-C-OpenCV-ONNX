
#include <iostream>
#include <cstdlib>
#include <cstdio>

#include <cstring>
#include <fstream>
#include <sstream>

#include <cmath> // Standard header library declares a set of mathematical functions
#include <random> // Random numbers

// OpenCV Headers
#include <opencv2/core/core.hpp> 
#include <opencv2/video/video.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/videoio/videoio.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/imgcodecs/imgcodecs.hpp> 

// DNN Module
#include <opencv2/dnn/dnn.hpp>


using namespace std;
using namespace cv;
using namespace dnn;

std::vector<std::string> class_names = {"manzana", "mango", "aguacate", "guineo", "tomate"};
cv::RNG rng(3);

cv::Mat prepare_input(cv::Mat image, int input_width, int input_height) {
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, Size(640,640), cv::Scalar(), true, false);
    return blob;
}

std::vector<cv::Rect> rescale_boxes(std::vector<cv::Rect> boxes, int img_width, int img_height) {
    std::vector<cv::Rect> rescaled_boxes;
    float scale_x = static_cast<float>(img_width) / 640.0;
    float scale_y = static_cast<float>(img_height) / 640.0;

    for (const auto& box : boxes) {
        int x1 = static_cast<int>(box.x * scale_x);
        int y1 = static_cast<int>(box.y * scale_y);
        int x2 = static_cast<int>((box.x + box.width) * scale_x);
        int y2 = static_cast<int>((box.y + box.height) * scale_y);

        rescaled_boxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }

    return rescaled_boxes;
}

std::vector<cv::Rect> xywh2xyxy(std::vector<cv::Rect> boxes) {
    std::vector<cv::Rect> converted_boxes;
    for (auto& box : boxes) {
        int x1 = box.x - box.width / 2;
        int y1 = box.y - box.height / 2;
        int x2 = box.x + box.width / 2;
        int y2 = box.y + box.height / 2;
        converted_boxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }
    return converted_boxes;
}

void drawdetect(cv::Mat& image, std::vector<cv::Rect> boxes, std::vector<float> scores, std::vector<int> class_ids, float mask_alpha = 0.3) {
    cv::Mat det_img = image.clone();
    int img_height = image.rows;
    int img_width = image.cols;
    float font_size = std::min(img_height, img_width) * 0.0006;
    int text_thickness = static_cast<int>(std::min(img_height, img_width) * 0.001);

    cout << "Total of detections = " << boxes[0] << endl;

    for (size_t i = 0; i < boxes.size(); ++i) {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::rectangle(det_img, boxes[i], color, 2);

       //  std::string label = class_names[class_ids[i]] + " " + std::to_string(static_cast<int>(scores[i] * 100)) + "%";
         cv::Point org(boxes[i].tl().x, boxes[i].tl().y - 10);
         cv::putText(det_img, "hola", org, cv::FONT_HERSHEY_SIMPLEX, font_size, cv::Scalar(255, 255, 255), text_thickness);
    // 
    }


    cv::imshow("Output", det_img);
    cv::waitKey(0);
}
    struct Detection
    {
        int class_id{0};
        std::string className{};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
    };

int main() {
    std::string model_path = "MRV.onnx";
    float modelNMSThreshold = 0.05;
    float modelScore = 0.1;


    // Initialize model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
    
    // Load image
    std::string img_url = "ma.jpeg";
    cv::Mat img = cv::imread(img_url);
    
    // Prepare input
    cv::Mat input_tensor = prepare_input(img, 640, 640);

    // Perform inference
    net.setInput(input_tensor);
    std::vector<int> output_names;
    output_names.push_back(267);
 
    vector<Mat> outputs;
    
    net.forward(outputs, net.getUnconnectedOutLayersNames());


    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);

    cv::transpose(outputs[0], outputs[0]);

    float *data = (float *)outputs[0].data;

    float x_factor =static_cast<float>(img.cols) / 640;
    float y_factor = static_cast<float>(img.rows) / 640;

    // Process output
    // cv::Mat detections = outputs[0].reshape(1, outputs[0].total() / 8400);
    std::vector<cv::Rect> boxes;
    std::vector<float> conf;
    std::vector<int> class_ids;


    
     for (int i = 0; i < rows; ++i) {
         float *classes_scores = data+4;


        cv::Mat scores(1, class_names.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
       

          if (maxClassScore > modelScore) {
                std::cout << "Number of detections:" << maxClassScore << std::endl;
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x -  0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);



            boxes.push_back(cv::Rect(left, top, width, height));
            
            conf.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
        }

     
        data += dimensions;

        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, conf, modelScore, modelNMSThreshold, nms_result);

        std::vector<Detection> detections{};

         for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = conf[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = class_names[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    int dt = detections.size();
    std::cout << "Number of detections:" << dt << std::endl;

    Mat frame=img.clone();
     for (int i = 0; i < dt; ++i)
        {
            Detection New_detection = detections[i];

            cv::Rect box = New_detection.box;
            std::cout << "Number of detections:" << box << std::endl;
            cv::Scalar color = New_detection.color;

            // // Detection box
            std::cout << box  ;
            cv::rectangle(frame, box, color, 1);

            // // Detection box text
            std::string classString = New_detection.className + ' ' + std::to_string(New_detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);          
        }

        cv::imshow("Inference", frame);

        cv::waitKey(0);

        

     

    return 0;
}