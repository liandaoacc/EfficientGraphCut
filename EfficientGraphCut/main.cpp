#include <cstdio>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>

#include <boost/pending/disjoint_sets.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "efficientgraphcut.h"

//class Edge
//{
//public:
//    Edge() : vertex1(0), vertex2(0), weight(0) { }
//    Edge(int v1, int v2, float w) : vertex1(v1), vertex2(v2), weight(w) { }
//    ~Edge() { }
//    float weight;
//    int vertex1, vertex2;
//    bool operator< (const Edge& b) const { return weight < b.weight; }
//};
//
//void edgeTest()
//{
//    Edge edgeSet[4];
//    for (int i = 0; i < 4; i++)
//    {
//        edgeSet[i].vertex1 = i;
//        edgeSet[i].vertex2 = i + 1;
//        edgeSet[i].weight = edgeSet[i].vertex2 * edgeSet[i].vertex2 - edgeSet[i].vertex1 * edgeSet[i].vertex1;
//    }
//
//    std::sort(edgeSet, edgeSet + 4);
//}

void efficientGraphCutUnitDemo(cv::Mat& image, double sigma = 0.8, float kthrehold = 200, unsigned int minsize = 10)
{
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat imageBGR[3];
    cv::Mat segImageB, imageBGauss, segImageBCvt, segImageBMap;
    unsigned int numberOfComponent = 0;

    cv::split(image, imageBGR);

    EfficientGraphCutUnit gCut(imageBGR[0].cols, imageBGR[0].rows, sigma, kthrehold, minsize);
    gCut.generateGraph(imageBGR[0]);
    gCut.segmentGraph(segImageB);
    numberOfComponent = gCut.componentSize();
    cv::GaussianBlur(imageBGR[0], imageBGauss, cv::Size(), sigma, 0, cv::BORDER_REPLICATE);
    segImageB.convertTo(segImageBCvt, CV_8UC1, 255.0 / numberOfComponent);
    cv::applyColorMap(segImageBCvt, segImageBMap, cv::COLORMAP_HSV);

    std::cout << "Number of component: " << numberOfComponent << std::endl;

    cv::imshow("Origin Image", imageBGR[0]);
    cv::imshow("Smoothed Image", imageBGauss);
    cv::imshow("Segment Result", segImageBMap);
    cv::waitKey(0);

    cv::destroyAllWindows();
}

void efficientGraphCutUnifyRGBDemo(cv::Mat& image, double sigma = 0.5, float kthrehold = 50, unsigned int minsize = 200)
{
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat segImage, imageGauss, segImageCvt, segImageMap;
    unsigned int numberOfComponent = 0;

    EfficientGraphCutUnifyRGB gCut(image.cols, image.rows, sigma, kthrehold, minsize);
    gCut.generateGraph(image);
    gCut.segmentGraph(segImage);
    numberOfComponent = gCut.componentSize();
    cv::GaussianBlur(image, imageGauss, cv::Size(), sigma, 0, cv::BORDER_REPLICATE);
    segImage.convertTo(segImageCvt, CV_8UC3, 255.0 / numberOfComponent);
    cv::applyColorMap(segImageCvt, segImageMap, cv::COLORMAP_HSV);

    std::cout << "Number of component (boost): " << numberOfComponent << std::endl;

    //cv::imshow("Origin Image", image);
    //cv::imshow("Smoothed Image", imageGauss);
    cv::imshow("Segment Result", segImageMap);
    cv::waitKey(0);

    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
    double sigma = 0.8;
    float kthrehold = 200;
    unsigned int minsize = 10;
    cv::String imageName = "77.jpg";
    const cv::String keys =
        "{help h usage ? |        | print this message                 }"
        "{@imageName     | 77.jpg | image to segment                   }"
        "{sigma          | 0.8    | Gaussian kernel standard deviation }"
        "{kthrehold      | 200    | constant for threshold function    }"
        "{minsize        | 10     | minimum component size             }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    sigma = parser.get< double >("sigma");
    kthrehold = parser.get< float >("kthrehold");
    minsize = parser.get< unsigned int >("minsize");
    imageName = parser.get< cv::String >(0);
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    cv::Mat image = cv::imread(imageName);
    efficientGraphCutUnitDemo(image, sigma, kthrehold, minsize);
    //efficientGraphCutUnifyRGBDemo(image, sigma, kthrehold, minsize);
    return 0;

}