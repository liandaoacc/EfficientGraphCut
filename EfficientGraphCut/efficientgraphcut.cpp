#include <vector>
#include <queue>
#include <map>
#include <functional>
#include <iterator>
#include <cmath>


#include <boost/graph/graph_concepts.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/pending/indirect_cmp.hpp>
#include <boost/concept/assert.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "efficientgraphcut.h"

void EfficientGraphCutBase::generateComponentMapImpl(Graph& g, cv::Mat& componentMap)
{
    BOOST_CONCEPT_ASSERT(( boost::VertexListGraphConcept< Graph > ));
    BOOST_CONCEPT_ASSERT(( boost::EdgeListGraphConcept< Graph > ));
    typedef boost::graph_traits< Graph >::vertex_descriptor Vertex;
    BOOST_CONCEPT_ASSERT(( boost::UnsignedInteger< Vertex > ));
    typedef boost::graph_traits< Graph >::edge_descriptor Edge;
    typedef boost::graph_traits< Graph >::vertices_size_type size_type;
    
    typedef boost::property_map< Graph, boost::vertex_index_t >::type vertex_index_map_t;
    typedef boost::property_map< Graph, boost::edge_weight_t >::type edge_weight_map_t;
    typedef boost::property_traits< edge_weight_map_t >::value_type W_value;
    //typedef boost::property_traits< edge_weight_map_t >::key_type W_key;
    //BOOST_CONCEPT_ASSERT(( boost::ReadablePropertyMapConcept< edge_weight_map_t, W_key > ));
    BOOST_CONCEPT_ASSERT(( boost::ReadablePropertyMapConcept< edge_weight_map_t, Edge > ));
    BOOST_CONCEPT_ASSERT(( boost::ComparableConcept< W_value > ));
    typedef boost::property_map< Graph, internaldiff_t >::type vertex_internaldiff_map_t;
    typedef boost::property_traits< vertex_internaldiff_map_t >::value_type D_value;
    BOOST_CONCEPT_ASSERT(( boost::ReadWritePropertyMapConcept< vertex_internaldiff_map_t, Vertex > ));
    BOOST_CONCEPT_ASSERT(( boost::ComparableConcept< D_value > ));
    typedef boost::property_map< Graph, componentsize_t >::type vertex_componentsize_map_t;
    typedef boost::property_traits< vertex_componentsize_map_t >::value_type C_value;
    BOOST_CONCEPT_ASSERT(( boost::ReadWritePropertyMapConcept< vertex_componentsize_map_t, Vertex > ));
    BOOST_CONCEPT_ASSERT(( boost::UnsignedInteger< C_value > ));

    size_type n = num_vertices(g);
    if (n == 0) return; // Nothing to do in this case
    
    typedef std::vector< size_type > Rank;                                                       
    typedef std::vector< Vertex > Parent;  
    typedef std::vector< size_type >::iterator RankIter;
    typedef std::vector< Vertex >::iterator ParentIter;
    typedef boost::iterator_property_map< RankIter, vertex_index_map_t, 
        std::iterator_traits< RankIter >::value_type, 
        std::iterator_traits< RankIter >::reference > RankMap;
    typedef boost::iterator_property_map< ParentIter, vertex_index_map_t, 
        std::iterator_traits< ParentIter >::value_type, 
        std::iterator_traits< ParentIter >::reference > ParentMap;
    typedef boost::property_traits<RankMap>::value_type R_value;
    typedef boost::property_traits<ParentMap>::value_type P_value;
    BOOST_CONCEPT_ASSERT(( boost::ConvertibleConcept<P_value, Vertex> ));
    BOOST_CONCEPT_ASSERT(( boost::IntegerConcept<R_value> ));

    Rank rank_map(n);
    Parent pred_map(n);

    boost::disjoint_sets < RankMap, ParentMap > dset(
        boost::make_iterator_property_map(rank_map.begin(), boost::get(boost::vertex_index, g), rank_map[0]),
        boost::make_iterator_property_map(pred_map.begin(), boost::get(boost::vertex_index, g), pred_map[0]));

    boost::graph_traits<Graph>::vertex_iterator ui, uiend;
    for (boost::tie(ui, uiend) = vertices(g); ui != uiend; ui++)
        dset.make_set(*ui);
    
    typedef boost::indirect_cmp< edge_weight_map_t, std::greater<W_value> > weight_greater;
    weight_greater wgreater(boost::get(boost::edge_weight, g));
    std::priority_queue<Edge, std::vector<Edge>, weight_greater> Q(wgreater);

    /*push all edge into Q*/
    boost::graph_traits<Graph>::edge_iterator ei, eiend;
    for (boost::tie(ei, eiend) = boost::edges(g); ei != eiend; ei++) 
        Q.push(*ei);

    while (! Q.empty()) 
    {
        Edge e = Q.top();
        Q.pop();
        Vertex u = dset.find_set(boost::source(e, g));
        Vertex v = dset.find_set(boost::target(e, g));
        W_value w = get(boost::edge_weight, g, e);

        if ( u != v ) 
        {
            if( w <= get(internaldiff_t(), g, u) && w <= get(internaldiff_t(), g, v))
            {
                C_value csize = get(componentsize_t(), g, u) + get(componentsize_t(), g, v);
                dset.link(u, v);
                Vertex re = dset.find_set(u);
                
                boost::put(componentsize_t(), g, re, csize);
                
                float newdiff = w + threholdFunction(csize);
                boost::put(internaldiff_t(), g, re, newdiff);
            }
        }
    }

    // post process small components
    for (boost::tie(ei, eiend) = boost::edges(g); ei != eiend; ei++) 
    {
        Vertex u = dset.find_set(boost::source(*ei, g));
        Vertex v = dset.find_set(boost::target(*ei, g));
        C_value usize = get(componentsize_t(), g, u);
        C_value vsize = get(componentsize_t(), g, v);
        if ((u != v) && ((usize < minComponentSize) || (vsize < minComponentSize)))
        {
            dset.link(u, v);
            Vertex re = dset.find_set(u);
            boost::put(componentsize_t(), g, re, usize + vsize);
        }
    }

    boost::tie(ui, uiend) = vertices(g);
    numOfComponent = dset.count_sets(ui, uiend);

    std::map< Vertex, int > marker;
    int componentIndex = 0;
    componentMap.create(height, width, CV_32S);

    for (boost::tie(ui, uiend) = vertices(g); ui != uiend; ui++)
    {
        Vertex u = *ui;
        Vertex re = dset.find_set(u);
        if (marker.find(re) == marker.end())
        {
            marker[re] = componentIndex++;
        }
        componentMap.at< int >(u / width, u % width) = marker[re];
    }
}

float EfficientGraphCutUnit::weight(cv::Mat& image, const int x1, const int y1, const int x2, const int y2)
{
    const float& p1 = image.at< float >(y1, x1);
    const float& p2 = image.at< float >(y2, x2);
    return std::abs(p1 - p2);
}

void EfficientGraphCutUnit::generateGraph(cv::Mat& image)
{
    CV_Assert(image.type() == CV_8UC1);

    cv::Mat imageGauss, imageGaussf;
    cv::GaussianBlur(image, imageGauss, cv::Size(), sigma, 0, cv::BORDER_REPLICATE);
    imageGauss.convertTo(imageGaussf, CV_32FC1, 1.0/255);
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            if (x < width-1)
            {
                unsigned int a = y * width + x;
                unsigned int b = y * width + (x + 1);
                float w = weight(imageGaussf, x, y, x + 1, y);
                add_edge(a, b, EdgeProperty(w), g);
            }
            if (y < height-1)
            {
                unsigned int a = y * width + x;
                unsigned int b = (y + 1) * width + x;
                float w = weight(imageGaussf, x, y, x, y + 1);
                add_edge(a, b, EdgeProperty(w), g);
            }
            if ((x < width-1) && (y < height-1))
            {
                unsigned int a = y * width + x;
                unsigned int b = (y + 1) * width + (x + 1);
                float w = weight(imageGaussf, x, y, x + 1, y + 1);
                add_edge(a, b, EdgeProperty(w), g);
            }
            if ((x < width-1) && (y > 0))
            {
                unsigned int a = y * width + x;
                unsigned int b = (y - 1) * width + (x + 1);
                float w = weight(imageGaussf, x, y, x + 1, y - 1);
                add_edge(a, b, EdgeProperty(w), g);
            }
        }
    }

    boost::property_map < Graph, internaldiff_t >::type internaldiffMap = boost::get(internaldiff_t(), g);
    boost::property_map < Graph, componentsize_t >::type componentsizeMap = boost::get(componentsize_t(), g);
    for (std::size_t i = 0; i < boost::num_vertices(g); i++)
    {
        internaldiffMap[i] = threholdFunction(1);
        componentsizeMap[i] = 1;
    }
}

void EfficientGraphCutUnit::segmentGraph(cv::Mat& segmentResult)
{
    generateComponentMapImpl(g, segmentResult);
}

float EfficientGraphCutUnifyRGB::weight(cv::Mat& image, const int x1, const int y1, const int x2, const int y2)
{
    const cv::Vec3f& p1 = image.at< cv::Vec3f >(y1, x1);
    const cv::Vec3f& p2 = image.at< cv::Vec3f >(y2, x2);
    return std::sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]) + (p1[2] - p2[2])*(p1[2] - p2[2]));
}

void EfficientGraphCutUnifyRGB::generateGraph(cv::Mat& image)
{
    CV_Assert(image.type() == CV_8UC3);

    cv::Mat imagef, imageGauss;
    image.convertTo(imagef, CV_32FC3, 1.0/255);
    cv::GaussianBlur(imagef, imageGauss, cv::Size(), sigma, 0, cv::BORDER_REPLICATE);

    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            if (x < width-1)
            {
                unsigned int a = y * width + x;
                unsigned int b = y * width + (x + 1);
                float w = weight(imageGauss, x, y, x + 1, y);
                add_edge(a, b, EdgeProperty(w), g);
            }
            if (y < height-1)
            {
                unsigned int a = y * width + x;
                unsigned int b = (y + 1) * width + x;
                float w = weight(imageGauss, x, y, x, y + 1);
                add_edge(a, b, EdgeProperty(w), g);
            }
            if ((x < width-1) && (y < height-1))
            {
                unsigned int a = y * width + x;
                unsigned int b = (y + 1) * width + (x + 1);
                float w = weight(imageGauss, x, y, x + 1, y + 1);
                add_edge(a, b, EdgeProperty(w), g);
            }
            if ((x < width-1) && (y > 0))
            {
                unsigned int a = y * width + x;
                unsigned int b = (y - 1) * width + (x + 1);
                float w = weight(imageGauss, x, y, x + 1, y - 1);
                add_edge(a, b, EdgeProperty(w), g);
            }
        }
    }

    std::cout << "Number of edges (boost): " << boost::num_edges(g) << std::endl;

    boost::property_map < Graph, internaldiff_t >::type internaldiffMap = boost::get(internaldiff_t(), g);
    boost::property_map < Graph, componentsize_t >::type componentsizeMap = boost::get(componentsize_t(), g);
    for (std::size_t i = 0; i < boost::num_vertices(g); i++)
    {
        internaldiffMap[i] = threholdFunction(1);
        componentsizeMap[i] = 1;
    }
}
