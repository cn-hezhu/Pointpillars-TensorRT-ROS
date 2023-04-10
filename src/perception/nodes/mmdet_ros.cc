// headers in STL
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>

// headers in PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>

// headers in ROS
#include <tf/transform_datatypes.h>
#include <ros/package.h>

// headers in local files
#include "autoware_msgs/DetectedObjectArray.h"
#include "mmdet_ros.h"

void load_anchors(float* &anchor_data, string file_name)
{
  ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  vector<float> temp_points;
  string c;

  while (!InFile.eof())
  {
      InFile >> c;

      temp_points.push_back(atof(c.c_str()));
  }
  anchor_data = new float[temp_points.size()];
  for (size_t i = 0 ; i < temp_points.size() ; ++i) {
      anchor_data[i] = temp_points[i];
  }
  InFile.close();
  return;
}

MMdetPPROS::MMdetPPROS() 
    : DB_CONF_("/home/nvidia/pointpillars_ws/src/perception/bootstrap.yaml"), NUM_POINT_FEATURE_(4), OUTPUT_NUM_BOX_FEATURE_(7), TRAINED_SENSOR_HEIGHT_(PP_SENSOR_HEIGHT), NORMALIZING_INTENSITY_VALUE_(255.0f)
{
    YAML::Node config = YAML::LoadFile(DB_CONF_);
    if(config["UseOnnx"].as<bool>()) {
        pfe_file_ = config["PfeOnnx"].as<std::string>();
        backbone_file_ = config["BackboneOnnx"].as<std::string>();
        ROS_INFO("UseOnnx");
    }else {
        pfe_file_ = config["PfeTrt"].as<std::string>();
        backbone_file_ = config["BackboneTrt"].as<std::string>();
        ROS_INFO("UseTrt");
    }
    std::cout << "pfe_file: " << pfe_file_ << std::endl;
    std::cout << "backbone_file: " << backbone_file_ << std::endl;
    const std::string pp_config_ = config["ModelConfig"].as<std::string>();

    point_pillars_ptr_.reset(new PointPillars(config["UseOnnx"].as<bool>(), pfe_file_, backbone_file_,pp_config_));
    ROS_INFO("finish init multihead_point_pillars...");
}

void MMdetPPROS::createROSPubSub()
{
  sub_points_ = nh_.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, &MMdetPPROS::pointsCallback, this);
  pub_objects_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);
}

void MMdetPPROS::pubDetectedObject(const std::vector<float> &detections, const std::vector<int> &labels, const std::vector<float> &scores, const std_msgs::Header &in_header)
{
  autoware_msgs::DetectedObjectArray objects;
  objects.header = in_header;
  objects.header.frame_id = "lidar";
  int num_objects = detections.size() / OUTPUT_NUM_BOX_FEATURE_;
  for (int i = 0; i < num_objects; i++)
  {
    autoware_msgs::DetectedObject object;
    object.header = in_header;
    object.valid = true;
    object.pose_reliable = true;
    // pose
    object.pose.position.x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 0];
    object.pose.position.y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 1];
    object.pose.position.z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 2];
    // yaw
    float yaw = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 6];
    yaw += M_PI / 2;
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));
    geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(-yaw);
    object.pose.orientation = q;
    // dimensions
    object.dimensions.x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 3];
    object.dimensions.y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 4];
    object.dimensions.z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 5];
    // label
    if(labels[i] == 0 && scores[i] > 0.8) 
    {
      object.label = "car";
    }
    else if (labels[i] == 1 && scores[i] > 0.8)
    {
      object.label = "Truck";
    }
    else if (labels[i] == 2 && scores[i] > 0.8)
    {
      object.label = "Van";
    }
    else if (labels[i] == 3 && scores[i] > 0.8)
    {
      object.label = "Bus";
    }
    // else if (labels[i] == 3 && scores[i] > 0.6)
    // {
    //   object.label = "Pedestrian";
    // }
    // else if (labels[i] == 4 && scores[i] > 0.8)
    // {
    //   object.label = "Tram";
    // }
    else
    {
      continue;
    }

    objects.objects.push_back(object);
    ROS_INFO("detect label %d , score %f ...", labels[i], scores[i]);
    ROS_INFO("direct detect object center %.2f %.2f %.2f", \
              object.pose.position.x, object.pose.position.y, object.pose.position.z);
  }
  pub_objects_.publish(objects);
}

void MMdetPPROS::pclToArray(const pcl::PointCloud<pcl::PointXYZI>::Ptr &in_pcl_pc_ptr, float *out_points_array,
                                 const float offset_z)
{
  for (size_t i = 0; i < in_pcl_pc_ptr->size(); i++)
  {
    pcl::PointXYZI point = in_pcl_pc_ptr->at(i);
    out_points_array[i * NUM_POINT_FEATURE_ + 0] = point.x;
    out_points_array[i * NUM_POINT_FEATURE_ + 1] = point.y;
    out_points_array[i * NUM_POINT_FEATURE_ + 2] = point.z + offset_z;
    out_points_array[i * NUM_POINT_FEATURE_ + 3] = float(point.intensity / NORMALIZING_INTENSITY_VALUE_);
  }

  // debug pcd points
  // ofstream ofFile;
  // ofFile.open("/home/nvidia/perception_ws/src/perception/test/data/testdata/pcd_points.txt" , std::ios::out );  
  // if (ofFile.is_open()) {
  //     for (int i = 0 ; i < in_pcl_pc_ptr->size() ; ++i) {
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 0] << " ";
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 1] << " ";
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 2] << " ";
  //         ofFile << out_points_array[i * NUM_POINT_FEATURE_ + 3] << " ";
  //         ofFile << "\n";
  //     }
  // }
  // ofFile.close();
}

void MMdetPPROS::pointsCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  ROS_INFO("pointsCallback");
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *pcl_pc_ptr);
  float *points_array = new float[pcl_pc_ptr->size() * NUM_POINT_FEATURE_];
  pclToArray(pcl_pc_ptr, points_array);

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  std::vector<float> out_detections;
  std::vector<int> out_labels;
  std::vector<float> out_scores;
  
  cudaDeviceSynchronize();
  point_pillars_ptr_->DoInference(points_array, pcl_pc_ptr->size(), &out_detections, &out_labels, &out_scores);
  cudaDeviceSynchronize();

  int num_objects = out_detections.size() / OUTPUT_NUM_BOX_FEATURE_;
  ROS_INFO("detect objects %d ...", num_objects);

  std::chrono::duration<double> time_elapse =
      std::chrono::steady_clock::now() - start;
  ROS_INFO("%.2lf ms", std::chrono::duration_cast<std::chrono::nanoseconds>(time_elapse)
                               .count() *
                           1.0e-6);

  delete[] points_array;
  pubDetectedObject(out_detections, out_labels, out_scores, msg->header);
}
