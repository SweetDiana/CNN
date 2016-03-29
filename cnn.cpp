#include "cnn.h"

Cnn::Cnn(void)
{
}

Cnn::~Cnn(void)
{
}

void Cnn::InitCnn()
{
	Layers vLayers1={'i',0,0,0};
	Layers vLayers2={'c',6,0,5};
	Layers vLayers3={'s',0,2,0};
	Layers vLayers4={'c',12,0,5};
	Layers vLayers5={'s',0,2,0};
	vLayers.push_back(vLayers1);
	vLayers.push_back(vLayers2);
	vLayers.push_back(vLayers3);
	vLayers.push_back(vLayers4);
	vLayers.push_back(vLayers5);
	
	nOpts_alpha=1;//学习率
	//每次挑出一个batchsize的batch来训练，也就是每用batchsize个样本就调整一次权值，而不是  
	//把所有样本都输入了，计算所有样本的误差了才调整一次权值  
	nOpts_batchsize=50;
	nOpts_numepochs=1;//训练次数，用同样的样本集
}

void Cnn::CnnSetup(vector<Mat>& vTrain_x,Mat& mTrain10_y)
{
	int nInputmaps=1;
	double fan_out,fan_in;  
    //取第一个图像样本后，再把维移除，就变成了28x28的矩阵，也就是行数与列数
	Mat mMapsize(1,2,CV_64F);
	mMapsize.at<double>(0,0)=vTrain_x[0].rows;
	mMapsize.at<double>(0,1)=vTrain_x[0].cols;
	//下面通过传入cnn_layers这个结构体来逐层构建CNN网络,size返回元素个数
	//CNN共有五层，这里范围的是5
	for(int l=0;l<vLayers.size();l++)
	{
		if(vLayers[l].cType=='s')//如果这层是 子采样层 
		{
			//subsampling层的mapsize，最开始mapsize是每张图的大小28*28
			//这里除以scale=2，pooling域之间没有重叠，所以pooling后的图像为14*14
			//注意这里的右边的mapsize保存的都是上一层每张特征map的大小，它会随着循环进行不断更新
			mMapsize=mMapsize/vLayers[l].nScale;
			for(int j=0;j<nInputmaps;j++)//inputmaps就是上一层有多少张特征图 
			{
				vLayers[l].vB.push_back(0);//将偏置初始化为0
			}
		}
		if(vLayers[l].cType=='c')//如果这层是 卷积层
		{
			//旧的mapsize保存的是上一层的特征map的大小，那么如果卷积核的移动步长是1，那用
			//kernelsize*kernelsize大小的卷积核卷积上一层的特征map后，得到的新的map的大小就是下面这样
			mMapsize=mMapsize-vLayers[l].nKernelsize+1;
			//该层需要学习的参数个数。每张特征map是一个(后层特征图数量)*(用来卷积的patch图的大小)  
            //因为是通过用一个核窗口在上一个特征map层中移动（核窗口每次移动1个像素），遍历上一个特征map  
            //层的每个神经元。核窗口由kernelsize*kernelsize个元素组成，每个元素是一个独立的权值，所以  
            //就有kernelsize*kernelsize个需要学习的权值，再加一个偏置值。另外，由于是权值共享，也就是  
            //说同一个特征map层是用同一个具有相同权值元素的kernelsize*kernelsize的核窗口去感受输入上一  
            //个特征map层的每个神经元得到的，所以同一个特征map，它的权值是一样的，共享的，权值只取决于  
            //核窗口。然后，不同的特征map提取输入上一个特征map层不同的特征，所以采用的核窗口不一样，也  
            //就是权值不一样，所以outputmaps个特征map就有（kernelsize*kernelsize+1）* outputmaps那么多的权值了  
            //但这里fan_out只保存卷积核的权值W，偏置b在下面独立保存
			fan_out=vLayers[l].nOutputmaps * vLayers[l].nKernelsize * vLayers[l].nKernelsize;
			
			Mat mRand_matrix(vLayers[l].nKernelsize,vLayers[l].nKernelsize,CV_64F);
			for(int j=0;j<vLayers[l].nOutputmaps;j++)
			{
				vLayers[l].vK.push_back(vector<Mat>(nInputmaps));
				//fan_out保存的是对于上一层的一张特征map，在这一层需要对这一张特征map提取outputmaps种特征，  
                //提取每种特征用到的卷积核不同，所以fan_out保存的是这一层输出新的特征需要学习的参数个数  
                //而，fan_in保存的是，在这一层，要连接到上一层中所有的特征map，然后用fan_out保存的提取特征  
                //的权值来提取他们的特征。也即是对于每一个当前层特征图，有多少个参数链到前层
				fan_in=nInputmaps * vLayers[l].nKernelsize * vLayers[l].nKernelsize;
				for(int i=0;i<nInputmaps;i++)//input map
				{
					//每次都用效率低不过格式调过来了
					//ConvNet.cnn_layers[l].layers_k[i].resize(ConvNet.cnn_layers[l].outputmaps);
					//随机初始化权值，也就是共有outputmaps个卷积核，对上层的每个特征map，都需要用这么多个卷积核去卷积提取特征。  
                    //产生n×n的 0-1之间均匀取值的数值的矩阵，再减去0.5就相当于产生-0.5到0.5之间的随机数,再 *2 就放大到 [-1, 1]。 
                    //就是将卷积核每个元素初始化为[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]  
                    //之间的随机数。因为这里是权值共享的，也就是对于一张特征map，所有感受野位置的卷积核都是一样的  
                    //所以只需要保存的是 inputmaps * outputmaps 个卷积核。
					randu(mRand_matrix,Scalar::all(0),Scalar::all(1));					
					vLayers[l].vK[j][i]=(mRand_matrix-0.5)*2*sqrt(6/(fan_in+fan_out));
				}
				vLayers[l].vB.push_back(0);//将偏置初始化为0 
			}
			//只有在卷积层的时候才会改变特征map的个数，pooling的时候不会改变个数。这层输出的特征map个数就是  
			nInputmaps=vLayers[l].nOutputmaps;//输入到下一层的特征map个数
		}
	}

	//fvnum 是输出层的前面一层的神经元个数,这一层的上一层是经过pooling后的层，
	//包含有inputmaps个特征map。每个特征map的大小是mapsize
	double dFvnum=mMapsize.at<double>(0,0)*mMapsize.at<double>(0,1)*nInputmaps;
	//onum 是标签的个数，也就是输出层神经元的个数。你要分多少个类，自然就有多少个输出神经元
	int nOnum=mTrain10_y.rows;

	//这里是最后一层神经网络的设定  
    //ffb 是输出层每个神经元对应的基biases
	mFfb=Mat::zeros(nOnum,1,CV_64F);
	//ffW 输出层前一层 与 输出层 连接的权值，这两层之间是全连接的
	mFfW=Mat(nOnum,(int)dFvnum,CV_64F);
	randu(mFfW,Scalar::all(0),Scalar::all(1));
	mFfW=(mFfW-0.5)*2*sqrt(6/(double(nOnum)+dFvnum));
}

void Cnn::CnnTrain(vector<Mat>& vTrain_x,Mat& mTrain10_y)
{
	//Train_x.size() 训练样本个数
	double dNumbatches=vTrain_x.size()/nOpts_batchsize;
	//相当于求余 就相当于取其小数部分，如果为0，就是整数
	if((dNumbatches-(int)dNumbatches)>0.000001)
	{
		cout<<"numbatches not integer"<<endl;
		return;
	}

	//打乱kk 1~10000,在for循环中初始化并随机化
	vector<int> vKk(vTrain_x.size());

	vector<Mat> batch_x(nOpts_batchsize);
	Mat batch_y(mTrain10_y.rows,nOpts_batchsize,CV_64F);

	for(int i=0;i<nOpts_numepochs;i++)
	{
		cout<<"epoch "<<(i+1)<<"/"<<nOpts_numepochs<<endl;
		//返回[1, N]之间所有整数的一个随机的序列
		//这样就相当于把原来的样本排列打乱，再挑出一些样本来训练
		for(int j=0;j<vTrain_x.size();j++)
			vKk[j]=j;
		random_shuffle(vKk.begin(),vKk.end());

		for(int l=0;l<(int)dNumbatches;l++)
		{
			//test
			cout<<l<<endl;

			for(int batch=l*nOpts_batchsize,m=0;batch<(l+1)*nOpts_batchsize,m<nOpts_batchsize;batch++,m++)
			{
				//取出打乱顺序后的batchsize个样本和对应的标签
				batch_x[m]=vTrain_x[vKk[batch]];
				mTrain10_y.col(vKk[batch]).copyTo(batch_y.col(m));
			}
			//在当前的网络权值和网络输入下计算网络的输出 Feedforward
			CnnFf(batch_x);
			//得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值
			//（也就是那些卷积核的元素）的导数
			CnnBp(batch_y);
			//得到误差对权值的导数后，就通过权值更新方法去更新权值
			CnnApplygrads();
			if(vRL.size()==0)
				vRL.push_back(dL);//代价函数值，也就是误差值
			//保存历史的误差值
			vRL.push_back(0.99*vRL.at(vRL.size()-1)+0.01*dL);
			
			//释放 vA
			for(int layer=0;layer<vLayers.size();layer++)
			{
				vector<vector<Mat>>().swap(vLayers[layer].vA);
				vector<vector<Mat>>().swap(vLayers[layer].vD);
				vector<vector<Mat>>().swap(vLayers[layer].vDk);
			}
		}
	}
}
