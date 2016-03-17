#include "Cnn.h"

int main(int argc,char* argv[])
{
	srand((unsigned)time(0));//random seed
	
	time_t startTime, stopTime;
	startTime = clock();	

	//load train data and test data
	ifstream infile1("D://software/visual studio 2012/workstation/Cnn_016/Cnn_016/test_x", ios::in | ios::binary);
	int nRows, nCols;
	infile1>>nRows>>nCols;
	Mat mTest_x = Mat::zeros(nRows, nCols, CV_8UC1);
	for (int i = 0; i < nRows; i++){
		for (int j = 0; j < nCols; j++)
		{
			int temp;
			infile1>>temp;
			mTest_x.at<uchar>(i, j) = temp;
		}
	}
	infile1.close();

	ifstream infile2("D://software/visual studio 2012/workstation/Cnn_016/Cnn_016/test_y", ios::in);
	infile2>>nRows>>nCols;
	Mat mTest_y = Mat::zeros(nRows, nCols, CV_8UC1);
	for (int i = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++)
		{
			int temp;
			infile2>>temp;
			mTest_y.at<uchar>(i, j) = temp;
		}
	infile2.close();

	ifstream infile3("D://software/visual studio 2012/workstation/Cnn_016/Cnn_016/train_x", ios::in);
	infile3>>nRows>>nCols;
	Mat mTrain_x = Mat::zeros(nRows, nCols, CV_8UC1);
	for (int i = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++){
			int temp;
			infile3>>temp;
			mTrain_x.at<uchar>(i, j) = temp;
		}
	infile3.close();

	ifstream infile4("D://software/visual studio 2012/workstation/Cnn_016/Cnn_016/train_y", ios::in);
	infile4>>nRows>>nCols;
	Mat mTrain_y = Mat::zeros(nRows, nCols, CV_8UC1);
	for (int i = 0; i < nRows; i++)
		for (int j = 0; j < nCols; j++)
		{
			int temp;
			infile4>>temp;
			mTrain_y.at<uchar>(i, j) = temp;
		}
	infile4.close();

	vector<Mat> vTrain_x=Cnn::ReShape(mTrain_x,28,60000);//60000*784->60000*28*28
	vector<Mat> vTest_x=Cnn::ReShape(mTest_x,28,10000);//10000*784->10000*28*28

	mTrain_y=mTrain_y.t();//60000*1->1(0~9)*60000
	mTest_y=mTest_y.t();//10000*1->1*10000
	mTrain_y.convertTo(mTrain_y,CV_64F);
	mTest_y.convertTo(mTest_y,CV_64F);
	
	//class variable
	Cnn ConvNet;

	ConvNet.InitCnn();//initial cnn,including cnn_layers & opts

	ConvNet.CnnSetup(vTrain_x,mTrain_y);

	ConvNet.CnnTrain(vTrain_x,mTrain_y);

	ConvNet.CnnTest(vTest_x,mTest_y);
	
	stopTime = clock();
	ofstream outFile("PftSalTime.txt",ofstream::app);
	outFile<<"pftsalTime:   "<<(float)(stopTime - startTime)/CLOCKS_PER_SEC<<endl;

	return 0;
}
