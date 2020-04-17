#include "treeseg.h"


#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <algorithm> // std::max
#include <pcl/segmentation/segment_differences.h>

int main(int argc, char *argv[])
{

  float dist = atof(argv[1]);
  char *originalst = argv[2];
  char *finalst = argv[3];

  pcl::PCDReader reader;
  pcl::PCDWriter writer;
  std::stringstream ss;
  //
  std::cout << "Reading original: " << std::flush;
  pcl::PointCloud<PointTreeseg>::Ptr orig(new pcl::PointCloud<PointTreeseg>);
  reader.read(originalst, *orig);
  std::cout << "complete" << std::endl;

  std::cout << "Reading final: " << std::flush;
  pcl::PointCloud<PointTreeseg>::Ptr final(new pcl::PointCloud<PointTreeseg>);
  reader.read(finalst, *final);
  std::cout << "complete" << std::endl;

  pcl::SegmentDifferences<PointTreeseg> sd;
  sd.setInputCloud(orig);
  sd.setDistanceThreshold(dist);

  // Set the target as itself
  sd.setTargetCloud(final);

  pcl::PointCloud<PointTreeseg> output;
  sd.segment(output);

  ss.str("");
  ss << "difference.pcd";

  writer.write(ss.str(), output, true);

  return 0;
}