//Andrew Burt - a.burt@ucl.ac.uk

#include "leafsep.h"

#include <pcl/io/pcd_io.h>

#include <armadillo>

int main (int argc, char* argv[])
{



	float smoothness = atof(argv[1]);
	bool sepwoodleaf = atoi(argv[2]);
	int start_argc = 3 ;


	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	std::stringstream ss;
	for(int i=start_argc;i<argc;i++)
	{
		std::cout << "----------: " << argv[i] << std::endl;
		std::cout << "Reading volume cloud: " << std::flush;
		std::vector<std::string> id = getFileID(argv[i]);
		pcl::PointCloud<PointTreeseg>::Ptr volume(new pcl::PointCloud<PointTreeseg>);
		reader.read(argv[i],*volume);
		std::cout << "complete" << std::endl;
		//
		std::cout << "Euclidean clustering: " << std::flush;
		std::vector<std::vector<float>> nndata = dNNz(volume,50,2);
		float nnmax = 0;
		for(int i=0;i<nndata.size();i++) if(nndata[i][1] > nnmax) nnmax = nndata[i][1];
		std::cout << nnmax << ", " << std::flush;
		std::vector<pcl::PointCloud<PointTreeseg>::Ptr> clusters;
		euclideanClustering(volume,nnmax,3,clusters);
		ss.str("");
		ss << id[0] << ".ec." << id[1] << ".pcd";
		writeClouds(clusters,ss.str(),false);
		std::cout << ss.str() << std::endl;
		//	
		std::cout << "Region-based segmentation: " << std::flush;
		std::vector<pcl::PointCloud<PointTreeseg>::Ptr> regions;
		int idx = findPrincipalCloudIdx(clusters);
		int nnearest = 50;
		int nmin = 3;
		regionSegmentation(clusters[idx],nnearest,nmin,smoothness,regions);
		ss.str("");
		ss << id[0] << ".ec.rg." << id[1] << ".pcd";
		writeClouds(regions,ss.str(),false);
		std::cout << ss.str() << std::endl;
		//
		if(sepwoodleaf == true)
		{
			std::cout << "Leaf stripping: " << std::endl;
			//
			std::cout << " Region-wise, " << std::flush;
		        arma::mat rfmat;
		        arma::gmm_full rmodel;
		        gmmByCluster(regions,5,1,5,50,100,rfmat,rmodel);
		        std::vector<int> rclassifications;
		        rclassifications = classifyGmmClusterModel(regions,5,rfmat,rmodel);
		        std::vector<pcl::PointCloud<PointTreeseg>::Ptr> csepclouds;
		        separateCloudsClassifiedByCluster(regions,rclassifications,csepclouds);
			ss.str("");
			ss << id[0] << ".ec.rg.rlw." << id[1] << ".pcd";
			writeCloudClassifiedByCluster(regions,rclassifications,ss.str());
			std::cout << ss.str() << std::endl;
			//
			std::cout << " Point-wise, " << std::flush;
			arma::mat pfmat;
			arma::gmm_diag pmodel;
			gmmByPoint(csepclouds[1],50,5,1,5,50,100,pfmat,pmodel);
			std::vector<int> pclassifications;
			pclassifications = classifyGmmPointModel(csepclouds[1],5,pfmat,pmodel);
			std::vector<pcl::PointCloud<PointTreeseg>::Ptr> psepclouds;
			separateCloudsClassifiedByPoint(csepclouds[1],pclassifications,psepclouds);
			ss.str("");
			ss << id[0] << ".ec.rg.rlw.plw." << id[1] << ".pcd";
			writeCloudClassifiedByPoint(csepclouds[1],pclassifications,ss.str());
			std::cout << ss.str() << std::endl;
			//
			ss.str("");
			ss << id[0] << ".ec.rg.rlw.plw.w." << id[1] << ".pcd";
			pcl::PointCloud<PointTreeseg>::Ptr wood(new pcl::PointCloud<PointTreeseg>);
			*wood += *csepclouds[0] + *psepclouds[0];
			writer.write(ss.str(),*wood,true);
			std::cout << ss.str() << std::endl;
			//
			std::cout << "Re-segmenting regions: " << std::flush;
			regions.clear();
			regionSegmentation(wood,nnearest,nmin,smoothness+2.5,regions);
			ss.str("");
			ss << id[0] << ".ec.rg.rlw.plw.w.rg." << id[1] << ".pcd";
			writeClouds(regions,ss.str(),false);
			std::cout << ss.str() << std::endl;
		}
		//
		std::cout << "Optimising regions: " << std::flush;
		removeFarRegions(regions);
		ss.str("");
		if(sepwoodleaf == true)	ss << id[0] << ".ec.rg.rlw.plw.w.rg.o." << id[1] << ".pcd";
		else ss << id[0] << ".ec.rg.o." << id[1] << ".pcd";
		writeClouds(regions,ss.str(),false);
		std::cout << ss.str() << std::endl;
		//
		std::cout << "Building tree: " << std::flush;
		pcl::PointCloud<PointTreeseg>::Ptr tree(new pcl::PointCloud<PointTreeseg>);
		buildTree(regions,tree);
		ss.str("");
		ss << id[0] << "_" << id[1] << ".pcd";
		writer.write(ss.str(),*tree,true);
		std::cout << ss.str() << std::endl;
	}
	return 0;
}
