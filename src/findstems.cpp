//Andrew Burt - a.burt@ucl.ac.uk

#include "treeseg.h"

#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <algorithm>    // std::max

const std::string reset("\033[0m");
const std::string red("\033[0;31m");
const std::string green("\033[0;32m");
const std::string cyan("\033[0;96m");


int main(int argc, char *argv[])
{

	float smoothness = atof(argv[1]);

	float dmin = atof(argv[2]);
	float dmax = atof(argv[3]);

	float lmin = atof(argv[4]) * 0.75; //assuming 3m slice == 3/4 slice h

	char* coordfileneame = argv[5];
	char* slicefileneame = argv[6];


	std::ifstream coordfile;
	std::cout << "coordfile: " << coordfileneame << std::endl;

	coordfile.open(coordfileneame);

	float coords[4] = {0};
	int n = 0;
	if (coordfile.is_open())
	{
		while (!coordfile.eof())
		{
			coordfile >> coords[n];

			n++;
		}
	}
	coordfile.close();
	float xmin = coords[0];
	float xmax = coords[1];
	float ymin = coords[2];
	float ymax = coords[3];

	std::cout << xmin << " " << xmax << " " << ymin << " " << ymax << std::endl;

	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	std::stringstream ss;
	//
	std::cout << "Reading slice: " << std::flush;
	std::vector<std::string> id = getFileID(slicefileneame);
	pcl::PointCloud<PointTreeseg>::Ptr slice(new pcl::PointCloud<PointTreeseg>);
	reader.read(slicefileneame, *slice);
	std::cout << "complete" << std::endl;
	//
	std::cout << "Cluster extraction: " << std::flush;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> clusters;
	int nnearest = 18;
	int nmin = 100;
	std::vector<float> nndata = dNN(slice, nnearest);
	euclideanClustering(slice, nndata[0], nmin, clusters);
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.pcd";
	writeClouds(clusters, ss.str(), false);
	std::cout << ss.str() << " | " << clusters.size() << std::endl;
	//
	std::cout << "Region-based segmentation: " << std::flush;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> regions;
	nnearest = 9;
	nmin = 100;
	for (int i = 0; i < clusters.size(); i++)
	{
		std::vector<pcl::PointCloud<PointTreeseg>::Ptr> tmpregions;
		regionSegmentation(clusters[i], nnearest, nmin, smoothness, tmpregions);
		for (int j = 0; j < tmpregions.size(); j++)
			regions.push_back(tmpregions[j]);
	}
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.pcd";
	writeClouds(regions, ss.str(), false);
	std::cout << ss.str() << " | " << regions.size() << std::endl;
	//
	std::cout << "RANSAC cylinder fits: " << std::flush;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> cyls;
	nnearest = 60;

	float stepcovmax = 0.1;
	float radratiomin = 0.8;
	float maxangle = 0.3;

	int cylnum = 0;
	for (int i = 0; i < regions.size(); i++)
	{
		std::cout << "Region : " << i << "\t of : " << regions.size() << std::flush;

		cylinder cyl;

		// if(i%10 ==0)
		// {
		fitCylinder(regions[i], nnearest, true, true, cyl);

		if (cyl.ismodel == true)
		{
			if (cyl.rad * 2 >= dmin && cyl.rad * 2 <= dmax && cyl.len >= lmin)
			{

				std::cout << "  cov: " << cyl.stepcov << std::flush;
				std::cout << "  max_dir: " << std::max(abs(cyl.dx),abs(cyl.dy)) << std::flush;

				if (cyl.stepcov <= stepcovmax)
				{
					if (cyl.radratio > radratiomin)
					{
						if ((cyl.x >= xmin && cyl.x <= xmax) && (cyl.y >= ymin && cyl.y <= ymax))
						{
							if ((abs(cyl.dx) < maxangle) && (abs(cyl.dy) < maxangle) )
							{
								cyls.push_back(cyl.inliers);
								std::cout << green << "  Added Cylinder : " << cylnum << reset <<std::flush;
								cylnum++;

							}
							else
							{
								std::cout << "  Bad angle" << std::flush;
							}
						}
						else
						{
							std::cout << "  Outside Coords :" << std::flush;
						}
					}
					else
					{
						std::cout << "  radratio too large," << std::flush;
					}
				}
				else
				{
					std::cout << "  cov too large," << std::flush;
				}
			}
		}
		// }
		std::cout << std::endl;
	}

	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.cylinders.pcd";
	writeClouds(cyls, ss.str(), false);
	std::cout << ss.str() << " | " << cyls.size() << std::endl;
	//
	std::cout << "Principal component trimming: " << std::flush;
	float anglemax = 35;
	std::vector<int> idx;
	for (int j = 0; j < cyls.size(); j++)
	{
		Eigen::Vector4f centroid;
		Eigen::Matrix3f covariancematrix;
		Eigen::Matrix3f eigenvectors;
		Eigen::Vector3f eigenvalues;
		computePCA(cyls[j], centroid, covariancematrix, eigenvectors, eigenvalues);
		Eigen::Vector4f gvector(eigenvectors(0, 2), eigenvectors(1, 2), 0, 0);
		Eigen::Vector4f cvector(eigenvectors(0, 2), eigenvectors(1, 2), eigenvectors(2, 2), 0);
		float angle = pcl::getAngle3D(gvector, cvector) * (180 / M_PI);
		if (angle >= (90 - anglemax) || angle <= (90 + anglemax))
			idx.push_back(j);
	}
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> pca;
	for (int k = 0; k < idx.size(); k++)
		pca.push_back(cyls[idx[k]]);
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.cylinders.principal.pcd";
	writeClouds(pca, ss.str(), false);
	std::cout << ss.str() << " | " << pca.size() << std::endl;
	//
	std::cout << "Concatenating stems: " << std::flush;
	float expansionfactor = 0;
	std::vector<pcl::PointCloud<PointTreeseg>::Ptr> stems;
	stems = pca;
	catIntersectingClouds(stems);
	ss.str("");
	ss << id[0] << ".intermediate.slice.clusters.regions.cylinders.principal.cat.pcd";
	writeClouds(stems, ss.str(), false);
	for (int m = 0; m < stems.size(); m++)
	{
		ss.str("");
		ss << id[0] << ".cluster." << m << ".pcd";
		writer.write(ss.str(), *stems[m], true);
	}
	std::cout << ss.str() << " | " << stems.size() << std::endl;
	//
	return 0;
}
