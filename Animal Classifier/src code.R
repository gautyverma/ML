#---------------------------------------------------------

#installing packages

#---------------instaling EBImage-------------------------  

#install.packages("BiocManager") 
#BiocManager::install("EBImage")
#library(EBImage) 

#----------------installing Tensorflow--------------------


#library(tensorflow) 
#install_tensorflow()

#---------installing keras and setting virtual-env------


#library(keras)
#install_keras()

# install using a conda environment (default is virtualenv)
#install_keras(method = "conda")

# install with GPU version of TensorFlow
# (NOTE: only do this if you have an NVIDIA GPU + CUDA!)
#install_keras(tensorflow = "gpu")

# install a specific version of TensorFlow
#install_keras(tensorflow = "1.2.1")
#install_keras(tensorflow = "1.2.1-gpu")

#---------------------------------------------------------
#-----C:\\Users\\GAUTY\\Desktop\\C-DAC\\dataset
setwd("C:\\Users\\GAUTY\\Downloads\\dataset")

library(keras)   
library(EBImage) 
library(tensorflow) 

train=list() 
test=list()

#List_OF_Images

#-------pic1-> Training dataset---------------

pic1=c("1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg",
       "8.jpg","9.jpg","10.jpg","11.jpg","12.jpg","13.jpg",
       "14.jpg","15.jpg","21.jpg","22.jpg","23.jpg","24.jpg",
       "25.jpg","26.jpg","27.jpg","28.jpg","29.jpg","30.jpg",
       "35.jpg","36.jpg","37.jpg","38.jpg","39.jpg")

#-------pic2-> Testing dataset---------------

pic2=c("16.jpg","17.jpg","18.jpg","19.jpg","20.jpg","31.jpg","32.jpg",
       "33.jpg","34.jpg","40.jpg")


for(i in 1:30){
   train[[i]]=readImage(pic1[i])
}

for(i in 1:10){
  test[[i]]=readImage(pic2[i])
}

#printing  and plotting

print(train[[1]])
display(train[[1]])
plot(train[[1]])

#Making a grid view of training and testing dataset 
#to view in  n rows and m columns we use "par" function

par(mfrow=c(5,6))
for(i in 1:30)
{
  plot(train[[i]])
}
for(i in 1:10)
{
  plot(test[[i]])
}

# resizing

# printing structure of images

str(train)
#------as the images are of different size and    ------- 
#------dimensions we must resize before processing-------

#resizing if training images 

for(i in 1:30)
{
  train[[i]]=resize(train[[i]],100,100)
}

#resizing of testing data

for(i in 1:10)
{
   test[[i]]=resize(test[[i]],100,100)
   
}

#checking the structure

str(train)
str(test)

train=combine(train)
x=tile(train,5)
display(x, title="train pics")

str(train)


test=combine(test)
y=tile(test,5)
display(y, title="train pics")
str(test)

# Re-order as " num [1:100, 1:100, 1:3, 1:25] " is not acceptable

#-----aperm() function transposes an array by permuting  
#-----its dimensions and optionally resizing it

train=aperm(train, c(4,1,2,3))
test=aperm(test,c(4,1,2,3))

 
#now check structure we get { num [1:25, 1:100, 1:100, 1:3] } ,this is acceptable
str(train)
str(test)

#-----assigning the values which to the dataset to predict desire output

trainy=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
         1,1,1,0,0,0,0,0,0,0,0,0,1,0,0)
testy=c(1,1,1,1,1,0,0,1,0,0)

# Hot Encoding//
# One hot encoding is a representation of categorical variables as binary vectors.

#----to_categorical is used to convert array of labeled data(from 0 to nb_classes-1) to one-hot vector

trainLables = to_categorical(trainy)
testLabels = to_categorical(testy)

#------------Traing the ML model-------------

#-----Pipe (%>%) Operator,this operator will forward a value or the
#-----result of an expression, into the next function call/expression.

model=keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32,kernel_size = c(3,3),activation = 'relu',input_shape = c(100,100,3))%>%
  layer_conv_2d(filters = 32,kernel_size = c(3,3),activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_dropout(rate = 0.25)%>%
  layer_conv_2d(filters = 64,kernel_size = c(3,3),activation = 'relu')%>%
  layer_conv_2d(filters = 64,kernel_size = c(3,3),activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_dropout(rate = 0.25)%>%
  layer_flatten()%>%
  layer_dense(units = 256,activation = 'relu')%>%
  layer_dropout(rate = 0.25)%>%
  layer_dense(units = 2 ,activation = 'softmax')%>%
  compile(loss ='categorical_crossentropy',optimizer = optimizer_sgd(lr=0.01,decay = 1e-6,momentum = 0.9,nesterov = T),metrics = c('accuracy'))

# summary of our model

summary(model)


history=model %>%
  fit(train,trainLables,epochs = 60,batch_size =30,validation_split = 0.2)

# epochs----An epoch is a term used in machine learning and indicates the number of 
# passes of the entire training dataset the machine learning algorithm has completed. 
  
#BATCH SIZE----Total number of training examples present in a single batch.

#------plotting the History graphs
#------that is lose and accuracy curves

plot(history)

model %>% evaluate(train,trainLables)
# shows the train model accuracy and loss 
pred = model %>% predict_classes(train,trainy,batch_size=30)

# training  data Confusion Matrix

table(predicted=pred,actual=trainy)


# probability of training data

pro = model %>% predict_proba(train,trainy,batch_size = 30)
cbind(pro,predicted=pred,actual=trainy)


#for testing data
model %>% evaluate(test,testLabels)

#probability of test data
pro1=model %>% predict_proba(test,testy,batch_size = 10)
cbind(pro1,predicted=pred,actual=testy)

pred_test=model %>%predict_classes(test,testy,batch_size = 10)
table(predicted=pred_test,actual=testy)


