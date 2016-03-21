#pragma once
#ifndef CNN_H
#define CNN_H

#include<iostream>
#include<vector>
#include<string>
#include<cstdlib>
#include<algorithm>
#include <fstream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<math.h>
#include<time.h>
#include<ctime>
#include"mnist.h"

using namespace cv;
using namespace std;

enum ConvolutionType {   
/* Return the full convolution, including border */
  CONVOLUTION_FULL, 
  
/* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,
  
/* Return only the submatrix containing elements that were not influenced by the border */
  CONVOLUTION_VALID
};
 
//声明cnn结构层
struct Layers{
	char cType;
	int nOutputmaps;
	int nScale;
	int nKernelsize;

	vector<double> vB;		//偏置
	vector<vector<Mat>> vK;	//卷积核 setup中初始化
	vector<vector<Mat>> vA;	//输出特征map	需要初始化
	vector<vector<Mat>> vD;	//灵敏度 或者 残差
	vector<vector<Mat>> vDk;
	vector<double> vDb;
};

class Cnn
{
public://类内包含的变量
	vector<Layers> vLayers;
	Mat mFfb;					//输出层每个神经元对应的基偏差
	Mat mFfW;					//输出层前一层 与 输出层 连接的权值
	int nOpts_alpha;			//学习率
	int nOpts_batchsize;		//每次挑出batchsize个数的batch来训练
	int nOpts_numepochs;		//训练次数
	vector<double> vRL;			//代价函数值也就是误差值
	double dL;					//代价函数是均方误差
	Mat mFv;					//最终提取到的特征向量
	Mat mO;						//网络的最终输出值
	Mat mE;						//error
	Mat mOd;					//输出层的灵敏度或残差
	Mat mFvd;					//残差反向传播回前一层
	Mat mDffW;
	Mat mDffb;

public://类内包含的函数
	Cnn(void);
	~Cnn(void);
	void InitCnn();//初始化cnn
	void CnnSetup(vector<Mat>&,Mat&);//构建网络
	void CnnTrain(vector<Mat>&,Mat&);
	void CnnFf(vector<Mat>&);
	void CnnBp(Mat&);
	void CnnApplygrads();
	void CnnTest(vector<Mat>&,Mat&);
	void CnnGetmaps(vector<Mat>&);
	static vector<Mat> ReShape(Mat&,int,int);//重塑数据矩阵
	static void Conv2(Mat&,Mat&,ConvolutionType,Mat&);//卷积操作
};

#endif
