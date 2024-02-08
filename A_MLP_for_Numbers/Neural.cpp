#include<eigen/Eigen/Eigen>
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include<Windows.h>
#include<iostream>

struct Inf
{
	char trainDataPath[50] = "model";
	char mnistTrainPath[50] = "MNIST/train-images.idx3-ubyte";
	char mnistTrainLablePath[50] = "MNIST/train-labels.idx1-ubyte";
	char mnistCheckPath[50] = "MNIST/train-images.idx3-ubyte";
	char mnistCheckLablePath[50] = "MNIST/train-labels.idx1-ubyte";

	const double learningRate = 0.005;

	int targetStep = 1e7;

	double cost = 0;

}runInf;

struct NeuralNetwork_Base
{
	Eigen::Matrix<double, 16, 784> m0 = Eigen::Matrix<double, 16, 784 >::Random(16,784);
	Eigen::Matrix<double, 16, 16> m1 = Eigen::Matrix<double, 16, 16 >::Random(16,16);
	Eigen::Matrix<double, 16, 16> m2 = Eigen::Matrix<double, 16, 16 >::Random(16,16);
	Eigen::Matrix<double, 10, 16> m3 = Eigen::Matrix<double, 10, 16 >::Random(10,16);
}MLP_WEIGHT;

struct NeuralNetwork_Data
{
	Eigen::Matrix<double, 784, 1> Out0 = Eigen::Matrix<double, 784, 1>::Zero();
	Eigen::Matrix<double, 16, 1> Out1 = Eigen::Matrix<double, 16, 1>::Zero();
	Eigen::Matrix<double, 16, 1> Out2 = Eigen::Matrix<double, 16, 1>::Zero();
	Eigen::Matrix<double, 16, 1> Out3 = Eigen::Matrix<double, 16, 1>::Zero();
	Eigen::Matrix<double, 10, 1> Out4 = Eigen::Matrix<double, 10, 1>::Zero();
	int lable = 0;
}MLP_FODATA,MLP_ERROR, MLP_DELTA;

struct ModelInf
{
	int epoch = 0, its = 0, image = 0;
}ModInf;

struct idxImage
{
	Eigen::Matrix<double, 784, 1> image = Eigen::Matrix<double, 784, 1>::Zero();
	int lable = 0;
}trainData[50];

void init();
void loop();
void train();
void clean();
void leave();
double check(int n);
void checkAll();
void forward();
void errorBackword();

void loadNetwork();
void saveNetwork();
void loadMnist();

int main()
{
	init();
	while (true)
	{
		loop();
	}
}

inline double activate(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

void init()
{
	int caseNum = 0;
	FILE* fp;
	fp = fopen(runInf.trainDataPath, "rb");
	if (fp == NULL)
	{
		caseNum = 1;
	}
	else
	{
		caseNum = 2;
		fclose(fp);
	}
	switch (caseNum)
	{
	case 1:
	{
		printf("No Model found, start from begin? y or n\n");
		char input;
		do
		{
			scanf("%1c", &input);
			clean();
			if (input == 'n' || input == 'N')
			{
				exit(2);
			}
			else if (input == 'y' || input == 'Y')
			{
				fp = fopen(runInf.trainDataPath, "wb+");
				fclose(fp);
				break;
			}
		} while (true);
		break;
	}
	case 2:
	{
		printf("One Model found, loading");
		Sleep(1500);
		loadNetwork();
		break;
	}
	}
}

void loop()
{
	int slectmode;
	system("cls");
	printf("1.train\n");
	printf("2.check\n");
	printf("3.check all the set\n");
	printf("4.leave\n");
	scanf("%d",&slectmode);
	clean();
	switch (slectmode)
	{
	case 1:
	{
		train();
		break;
	}
	case 2:
	{
		int n;
		printf("size: ");
		scanf("%d",&n);
		clean();
		printf("Rate is %.4lf",check(n));
		system("pause");
		break;
	}
	case 3:
	{
		checkAll();
		break;
	}
	case 4:
	{
		leave();
		break;
	}
	}
}

void train()
{
	int step = 0;
	printf("Steps: ");
	scanf("%d",&runInf.targetStep);

	while (runInf.targetStep>step)
	{
		loadMnist();
		for (int i = 0;i < 50;i++)
		{
			MLP_FODATA.Out0 = trainData[i].image;
			MLP_FODATA.lable = trainData[i].lable;
			forward();
			errorBackword();
			step++,ModInf.its++,ModInf.image++;
		}
		if (ModInf.image >= 60000)
		{
			ModInf.image = 0;
			ModInf.epoch++;
		}
		if (step % 5000 == 0)
		{
			double rate = check(100);
			printf("its: %d Rate: %.3lf Cost: %.4lf\n",ModInf.its,rate,runInf.cost);
			runInf.cost = 0;
		}
		if (step % 10000 == 0)
		{
			saveNetwork();
		}
	}
	saveNetwork();
}

void loadNetwork()
{
	FILE* fp = NULL;
	fp = fopen(runInf.trainDataPath,"rb+");
	if (fp == NULL)
	{
		exit(1);
	}
	fseek(fp,0,SEEK_SET);
	fread(&ModInf, sizeof(ModInf), 1, fp);
	fread(&MLP_WEIGHT,sizeof(MLP_WEIGHT),1,fp);
	fclose(fp);
}

void saveNetwork()
{
	FILE* fp;
	fp = fopen(runInf.trainDataPath, "wb+");
	fseek(fp, 0, SEEK_SET);
	fwrite(&ModInf, sizeof(ModInf), 1, fp);
	fwrite(&MLP_WEIGHT, sizeof(MLP_WEIGHT), 1, fp);
	fclose(fp);
}

inline void forward()
{
	MLP_FODATA.Out1 = MLP_WEIGHT.m0 * MLP_FODATA.Out0;
	for (int i = 0;i < 16;i++) { MLP_FODATA.Out1(i, 0) = activate(MLP_FODATA.Out1(i, 0)); }

	MLP_FODATA.Out2 = MLP_WEIGHT.m1 * MLP_FODATA.Out1;
	for (int i = 0;i < 16;i++) { MLP_FODATA.Out2(i, 0) = activate(MLP_FODATA.Out2(i, 0)); }

	MLP_FODATA.Out3 = MLP_WEIGHT.m2 * MLP_FODATA.Out2;
	for (int i = 0;i < 16;i++) { MLP_FODATA.Out3(i, 0) = activate(MLP_FODATA.Out3(i, 0)); }

	MLP_FODATA.Out4 = MLP_WEIGHT.m3 * MLP_FODATA.Out3;
	for (int i = 0;i < 10;i++) { MLP_FODATA.Out4(i, 0) = activate(MLP_FODATA.Out4(i, 0)); }
	/*
	printf(" %d: ", MLP_FODATA.lable);
	for (int i = 0;i < 10;i++)
	{
		printf("%.5lf ",MLP_FODATA.Out4(i,0));
	}
	printf("\n");
	*/
}

inline void errorBackword()
{
	for (int i = 0;i < 10;i++)
	{
		if (i == MLP_FODATA.lable)
		{
			MLP_ERROR.Out4(i, 0) = (0.99 - MLP_FODATA.Out4(i, 0));
		}
		else
		{
			MLP_ERROR.Out4(i, 0) = (0.01 - MLP_FODATA.Out4(i, 0));
		}
		runInf.cost += pow(MLP_ERROR.Out4(i, 0), 2);
		//printf("%.5lf ",MLP_ERROR.Out4(i,0));
	}
	//printf("\n\n");
	//Sleep(200);
	MLP_ERROR.Out3 = MLP_WEIGHT.m3.transpose() * MLP_ERROR.Out4;
	MLP_ERROR.Out2 = MLP_WEIGHT.m2.transpose() * MLP_ERROR.Out3;
	MLP_ERROR.Out1 = MLP_WEIGHT.m1.transpose() * MLP_ERROR.Out2;

	for (int i = 0;i < 10;i++) { MLP_DELTA.Out4(i, 0) = MLP_ERROR.Out4(i, 0) * MLP_FODATA.Out4(i, 0) * (1 - MLP_FODATA.Out4(i, 0)); }
	for (int i = 0;i < 16;i++) { MLP_DELTA.Out3(i, 0) = MLP_ERROR.Out3(i, 0) * MLP_FODATA.Out3(i, 0) * (1 - MLP_FODATA.Out3(i, 0)); }
	for (int i = 0;i < 16;i++) { MLP_DELTA.Out2(i, 0) = MLP_ERROR.Out2(i, 0) * MLP_FODATA.Out2(i, 0) * (1 - MLP_FODATA.Out2(i, 0)); }
	for (int i = 0;i < 16;i++) { MLP_DELTA.Out1(i, 0) = MLP_ERROR.Out1(i, 0) * MLP_FODATA.Out1(i, 0) * (1 - MLP_FODATA.Out1(i, 0)); }

	MLP_WEIGHT.m3 += runInf.learningRate * MLP_DELTA.Out4 * MLP_FODATA.Out3.transpose();
	MLP_WEIGHT.m2 += runInf.learningRate * MLP_DELTA.Out3 * MLP_FODATA.Out2.transpose();
	MLP_WEIGHT.m1 += runInf.learningRate * MLP_DELTA.Out2 * MLP_FODATA.Out1.transpose();
	MLP_WEIGHT.m0 += runInf.learningRate * MLP_DELTA.Out1 * MLP_FODATA.Out0.transpose();
}

void loadMnist()//load MNIST data to betch
{
	// images
	FILE* fp = NULL;
	fp = fopen(runInf.mnistTrainPath, "rb");
	if (fp == NULL)
	{
		printf("训练数据集异常");
		exit(1);
	}
	fseek(fp, 16, SEEK_SET);
	fseek(fp, ModInf.image * 784, SEEK_CUR);
	unsigned char image[784];
	for (int i = 0;i < 50;i++)
	{
		fread(image, 1, 784, fp);
		for (int j = 0;j < 784;j++)
		{
			trainData[i].image(j, 0) = ((image[j] / 255.0)*0.99)+0.01	;
		}
	}
	fclose(fp);
	// lables
	fp = fopen(runInf.mnistTrainLablePath, "rb");
	if (fp == NULL)
	{
		printf("训练数据集异常");
		exit(1);
	}
	fseek(fp, 8, SEEK_SET);
	fseek(fp, ModInf.image, SEEK_CUR);
	unsigned char lables[50];
	fread(lables, 1, 50, fp);
	for (int i = 0;i < 50;i++)
	{
		trainData[i].lable = lables[i];
	}
	fclose(fp);
}

double check(int n)
{
	double rate,max;
	int count=0,a;
	srand((unsigned)time(0));
	FILE* fp_1, * fp_2;
	fp_1 = fopen(runInf.mnistCheckPath,"rb");
	fp_2 = fopen(runInf.mnistCheckLablePath, "rb");
	unsigned char image[784];
	unsigned char lable;	
	if (fp_1 == NULL || fp_2 == NULL)
	{
		printf("校验数据异常");
		exit(1);
	}
	for (int i = 0;i < n;i++)
	{
		fseek(fp_1, 16, SEEK_SET);
		fseek(fp_2, 8, SEEK_SET);
		a = rand()%5000;
		fseek(fp_1, a * 784, SEEK_CUR);
		fseek(fp_2, a, SEEK_CUR);
		fread(image, 1, 784, fp_1);
		fread(&lable, 1, 1, fp_2);
		for (int j = 0;j < 784;j++)
		{
			MLP_FODATA.Out0(j, 0) = ((image[j] / 255.0)*0.99)+0.005;
		}
		forward();
		max=MLP_FODATA.Out4.maxCoeff();
		for (int i = 0;i < 10;i++)
		{
			if (MLP_FODATA.Out4(i, 0) == max)
			{
				if (i == lable)
				{
					count++;
				}
			}
		}
	}
	rate = count / (double)n;
	fclose(fp_1);
	fclose(fp_2);
	return rate;
}

void checkAll()
{
	double rate, max;
	int count = 0, a=0;
	FILE* fp_1, * fp_2;
	fp_1 = fopen(runInf.mnistCheckPath, "rb");
	fp_2 = fopen(runInf.mnistCheckLablePath, "rb");
	unsigned char image[784];
	unsigned char lable;
	if (fp_1 == NULL || fp_2 == NULL)
	{
		printf("校验数据异常");
		exit(1);
	}
	fseek(fp_1, 16, SEEK_SET);
	fseek(fp_2, 8, SEEK_SET);
	while (!feof(fp_1))
	{
		fread(image, 1, 784, fp_1);
		fread(&lable, 1, 1, fp_2);
		a++;
		for (int j = 0;j < 784;j++)
		{
			MLP_FODATA.Out0(j, 0) = ((image[j] / 255.0) * 0.99) + 0.005;
		}
		forward();
		max = MLP_FODATA.Out4.maxCoeff();
		for (int i = 0;i < 10;i++)
		{
			if (MLP_FODATA.Out4(i, 0) == max)
			{
				if (i == lable)
				{
					count++;
					break;
				}
			}
		}
		/*
		printf("%d  :",lable);
		for (int i = 0;i < 10;i++)
		{
			printf("%.5lf  ",MLP_FODATA.Out4(i,0));
		}
		printf("\n");
		Sleep(100);
		*/
	}
	rate = count / (double)a;
	fclose(fp_1);
	fclose(fp_2);
	printf("\n%d / %d\n", count, a);
	printf("\nRate is %lf\n",rate);
	system("pause");
	clean();
}

void leave()
{
	char c;
	system("cls");
	printf("Do you want to leave? y or n \n");
	c = getchar();
	if (c == 'y' || c == 'Y')
	{
		saveNetwork();
		exit(2);
	}
}

void clean()
{
	char c;
	while ((c = getchar()) != '\n' && c != EOF);
}
