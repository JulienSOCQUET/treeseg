//Andrew Burt - a.burt@ucl.ac.uk

#include "treeseg.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

float maxheight(float dbh)
{
	//m    -> 41.22 * dbh ^ 0.3406
	//ci_u -> 42.30 * dbh ^ 0.3697
	// float height = 42.30 * pow(dbh, 0.3697) + 5;
	// float height = 80 *dbh;

	dbh = dbh * 100.;
	float dpow = pow(dbh, 0.73);
	float height = (58.0 * dpow) / (21.8 + dpow) + 12;
	// https://doi.org/10.5194/bg-16-847-2019

	return height;
}

float maxcrown(float dbh)
{
	//m    -> 29.40 * dbh ^ 0.6524
	//ci_u -> 30.36 * dbh ^ 0.6931
	// float extent = 30.36 * pow(dbh, 0.6931) + 5;
	// float extent = 60 *dbh;

	dbh = dbh * 100.;
	float CA = 0.66 * pow(dbh, 1.34); //crown area
	// https://doi.org/10.5194/bg-16-847-2019

	float extent = 1.5 * (2.0 * sqrt(CA) / 1.772)+5; //sqrt pi // 1.5* safety factor

	return extent;
}

int main(int argc, char **argv)
{

	float diffmax = atof(argv[1]);//0.1

	int start_argc = 2;

	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	std::cout << "Reading plot-level cloud: " << std::flush;
	pcl::PointCloud<PointTreeseg>::Ptr plot(new pcl::PointCloud<PointTreeseg>);
	readTiles(argc, argv, plot);
	std::cout << "complete" << std::endl;

	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> volumes;

#pragma omp parallel
	{
#pragma omp for schedule(dynamic, 1)
		for (int i = start_argc; i < getTilesStartIdx(argc, argv); i++)
		{
			std::cout << "---------------" << std::endl;
			//
			std::vector<std::string> id = getFileID(argv[i]);
			pcl::PointCloud<PointTreeseg>::Ptr stem(new pcl::PointCloud<PointTreeseg>);
			reader.read(argv[i], *stem);
			//
			std::cout << "Estimating DBH: " << std::flush;
			int nnearest = 90;
			float zstep = 0.75;
			treeparams params = getTreeParams(stem, nnearest, zstep, diffmax);
			std::cout << params.d << std::endl;
			//
			std::cout << "Crown dimensions: " << std::flush;
			// if (params.d <0.15)
			// 	params.d =0.4;

			float h = maxheight(params.d);
			float c = maxcrown(params.d);

			std::cout << h << "m x " << c << "m (HxW)" << std::endl;
			//
			std::cout << "Segmenting volume: " << std::flush;
			pcl::PointCloud<PointTreeseg>::Ptr xslice(new pcl::PointCloud<PointTreeseg>);
			pcl::PointCloud<PointTreeseg>::Ptr yslice(new pcl::PointCloud<PointTreeseg>);
			pcl::PointCloud<PointTreeseg>::Ptr zslice(new pcl::PointCloud<PointTreeseg>);
			pcl::PointCloud<PointTreeseg>::Ptr volume(new pcl::PointCloud<PointTreeseg>);
			Eigen::Vector4f min, max, centroid;
			pcl::getMinMax3D(*stem, min, max);
			pcl::compute3DCentroid(*stem, centroid);
			spatial1DFilter(plot, "x", centroid[0] - c / 2, centroid[0] + c / 2, xslice);
			spatial1DFilter(xslice, "y", centroid[1] - c / 2, centroid[1] + c / 2, yslice);
			spatial1DFilter(yslice, "z", max[2], min[2] + h, zslice);
			std::cout << "z1: " << max[2] << "z2: " << min[2] + h << std::endl;

			*volume += *stem;
			*volume += *zslice;

			cylinder cyl;

			cyl.rad = c / 2.0;	 // max crown
			cyl.x = centroid[0]; // x position
			cyl.y = centroid[1]; // y position
			cyl.z = min[2];		 // ground level
			cyl.dx = 0;			 // vertical
			cyl.dy = 0;			 // vertical
			cyl.dz = 1;			 // vertical

			pcl::PointCloud<PointTreeseg>::Ptr volumecyl(new pcl::PointCloud<PointTreeseg>);

			std::cout << "Start Spatial3DCylinderFilter : " << std::endl;
			spatial3DCylinderFilter(volume, cyl, volumecyl);
			std::cout << "Done Spatial3DCylinderFilter : " << std::endl;

			std::stringstream ss;
			ss << id[0] << ".volume." << id[1] << ".pcd";
			writer.write(ss.str(), *volumecyl, true);
			std::cout << ss.str() << std::endl;
			// #pragma omp critical
			// 		{
			// 			volumes.push_back(volume);
			// 		}
		}
	}

	// std::stringstream ss;

	// std::vector<std::string> id = getFileID(argv[start_argc]);
	// ss.str("");
	// ss << id[0] << ".all_volums.pcd";
	// writeClouds(volumes, ss.str(), false);

	return 0;
}
