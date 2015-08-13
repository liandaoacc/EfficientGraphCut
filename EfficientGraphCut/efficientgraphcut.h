#ifndef EFFICIENT_GRAPH_CUT_H
#define EFFICIENT_GRAPH_CUT_H

#include <boost/graph/adjacency_list.hpp>

#include <opencv2/core.hpp>

class EfficientGraphCutBase
{
public:

    EfficientGraphCutBase(unsigned int imageWidth, unsigned int imageHeight, double sigmap, float kp, int minComponentSizep) 
        : width(imageWidth), height(imageHeight), sigma(sigmap), k(kp), minComponentSize(minComponentSizep), numOfComponent(0) { }
    ~EfficientGraphCutBase() { }

    struct internaldiff_t 
    {
        typedef boost::vertex_property_tag kind;
    };
    struct componentsize_t 
    {
        typedef boost::vertex_property_tag kind;
    };

    typedef boost::property< internaldiff_t, float, boost::property< componentsize_t, unsigned int> > VertexProperty;
    typedef boost::property< boost::edge_weight_t, float > EdgeProperty;
    typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, EdgeProperty > Graph;

    virtual float weight(cv::Mat& image, const int x1, const int y1, const int x2, const int y2) = 0;
    virtual float threholdFunction(const int sizeOfComponent) { return k/sizeOfComponent; }
    virtual void generateGraph(cv::Mat& image) = 0;
    virtual void segmentGraph(cv::Mat& segmentResult) = 0;
    virtual unsigned int componentSize() { return numOfComponent; }
    void generateComponentMapImpl(Graph& g, cv::Mat& componentMap);
protected:
    unsigned int width;
    unsigned int height;
    double sigma;
    float k;
    unsigned int minComponentSize;
    unsigned int numOfComponent;
};

class EfficientGraphCutUnit : public EfficientGraphCutBase
{
public:
    EfficientGraphCutUnit(unsigned int imageWidth, unsigned int imageHeight, double sigmap, float kp, int minComponentSizep)
        : EfficientGraphCutBase(imageWidth, imageHeight, sigmap, kp, minComponentSizep) { }
    ~EfficientGraphCutUnit() { }
    virtual float weight(cv::Mat& image, const int x1, const int y1, const int x2, const int y2);
    virtual void generateGraph(cv::Mat& image);
    virtual void segmentGraph(cv::Mat& segmentResult);

protected:
    Graph g;
};

class EfficientGraphCutUnifyRGB : public EfficientGraphCutUnit
{
public:
    EfficientGraphCutUnifyRGB(unsigned int imageWidth, unsigned int imageHeight, double sigmap, float kp, int minComponentSizep)
        : EfficientGraphCutUnit(imageWidth, imageHeight, sigmap, kp, minComponentSizep) { }
    ~EfficientGraphCutUnifyRGB() { }
    virtual float weight(cv::Mat& image, const int x1, const int y1, const int x2, const int y2);
    virtual void generateGraph(cv::Mat& image);
};
#endif