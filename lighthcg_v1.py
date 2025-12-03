# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,InputLayer,Conv2DTranspose,UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.initializers import HeNormal, GlorotNormal

tf.__version__ #implemented 2.19.0

import kagglehub

kagglehub.__version__ #0.3.13

os.environ['KAGGLEHUB_CACHE'] = '/tmp/kagglehub_cache'

import kagglehub

# Download latest version
path = kagglehub.dataset_download("orvile/acrima-glaucoma-assessment-using-fundus-images")

print("Path to dataset files:", path)

tr_dir = '/tmp/kagglehub_cache/datasets/orvile/acrima-glaucoma-assessment-using-fundus-images/versions/1/Database/Images'

#If already downloaded, set directory as below
#tr_dir = '/kaggle/input/acrima-glaucoma-assessment-using-fundus-images/Database/Images'

Filename = pd.DataFrame(os.listdir(tr_dir),columns=["filename"])
Filename

Filename = Filename.sort_values(by=['filename'], ascending=True)

Filename.index = [i for i in range(0,705,1)]

Filename

def load_images_and_labels(image,label):
    IMG = []
    LB = []
    for index, row in label.iterrows():
        img_path = os.path.join(image, f"{row['filename']}")
        if os.path.exists(img_path):
            img = cv2.imread(img_path,cv2.IMREAD_COLOR_RGB)
            img = cv2.resize(img, (224, 224)) # Resize images
            IMG.append(img)
            if "_g_" in row['filename']:
              LB.append(1)
            else:
              LB.append(0)
            if len(LB)%100 == 0: print(index+1)
    return np.array(IMG), np.array(LB)

# Loading images and labels
X_1,tag1 = load_images_and_labels(tr_dir,Filename)

np.bincount(tag1)

#Example of dataset
plt.imshow(X_1[1,:,:,:])
print(tag1[1])

###Train/test data

X_1 = X_1/255.0 #rescaling

#reshuffling data
(X_tr,X_te3,tag_tr,tag_te) = train_test_split(X_1,tag1,test_size=0.5,shuffle=True,random_state=321)
print(np.bincount(np.int32(tag_tr)))

plt.imshow(X_tr[5,:,:,:])
print(tag_tr[5])

#################Defining MRI_model

#Latent VAE + GAE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


tf.keras.utils.set_random_seed(321)

class lighthcg(tf.keras.Model):
  def __init__(self,input_dim, latent_dim1,latent_dim2,h1,h2,h3,h4,d1,d1_2,d2):
    super(lighthcg, self).__init__()
    self.input_dim = input_dim
    self.latent_dim1 = latent_dim1
    self.latent_dim2 = latent_dim2
    self.h1 = h1
    self.h2 = h2
    self.h3 = h3
    self.h4 = h4
    self.d1 = d1
    self.d1_2 = d1_2
    self.d2 = d2


    k = tf.keras.initializers.HeNormal(123)
    self.Enc = Sequential([
        InputLayer(shape=(input_dim,input_dim,3)),
        Conv2D(filters=h1,kernel_size=4,strides=(2,2),activation="silu"),
        Conv2D(filters=h2,kernel_size=6,strides=(3,3),activation="silu"),
        Conv2D(filters=h3,kernel_size=4,strides=(2,2),activation="silu"),
        Conv2D(filters=h4,kernel_size=3,strides=(2,2),activation="silu"),
        Flatten(),
        Dense(units = d1,activation="elu",kernel_initializer=k),
        Dense(units = d1_2,activation="elu",kernel_initializer=k),
        Dense(units=(latent_dim1+latent_dim2)*2,kernel_initializer=k,activation="linear")
        ])

    self.Dec = Sequential([
        InputLayer(shape=(latent_dim1+latent_dim2,)),
        Dense(units=d1_2, activation="elu",kernel_initializer=k),
        Dense(units = d1,activation="elu",kernel_initializer=k),
        Dense(units=8*8*h4, activation="elu",kernel_initializer=k),
        tf.keras.layers.Reshape(target_shape=(8,8,h4)),
        Conv2DTranspose(filters=h3, kernel_size=3, strides=(2,2), activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=h2, kernel_size=4, strides=(2,2), activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=h1, kernel_size=6, strides=(3,3),activation='silu',kernel_initializer=k),
        Conv2DTranspose(filters=3, kernel_size=4, strides=(2,2), kernel_initializer=k,activation='sigmoid'),
        ])



    mat1 = np.zeros((latent_dim2+1)*(latent_dim2+1)*d2).reshape(latent_dim2+1,(latent_dim2+1)*d2)
    mat2 = np.zeros((latent_dim2+1)*d2*(latent_dim2+1)*d2).reshape((latent_dim2+1)*d2,(latent_dim2+1)*d2)
    mat3 = np.zeros((latent_dim2+1)*d2*(latent_dim2+1)).reshape((latent_dim2+1)*d2,(latent_dim2+1))
    mat0 = np.ones((latent_dim2+1,latent_dim2+1)).reshape(latent_dim2+1,-1)

    for i in range(latent_dim2+1):
      mat1[i,(d2*i):(d2*(i+1))] = 1
      mat2[(i*d2):((i+1)*d2),(d2*i):(d2*(i+1))] = 1
      mat3[(i*d2):((i+1)*d2),i] = 1
      mat0[i,i] = 0

    mat0[-1,:] = 0

    mask1 = np.array(mat1).reshape(latent_dim2+1,(latent_dim2+1)*d2)
    class mask_1(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask1)*w


    mask2 = np.array(mat2).reshape((latent_dim2+1)*d2,(latent_dim2+1)*d2)
    class mask_2(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask2)*w

    mask3 = np.array(mat3).reshape((latent_dim2+1)*d2,latent_dim2+1)
    class mask_3(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask3)*w

    mask0 = np.array(mat0).reshape(latent_dim2+1,latent_dim2+1)
    class mask_0(tf.keras.constraints.Constraint):
      def __call__(self,w):
        return tf.convert_to_tensor(mask0)*w


    k2 = tf.keras.initializers.HeNormal(123)
    self.G_ENC = Sequential([
        InputLayer(shape=(latent_dim2+1,)),
        Dense(units = (latent_dim2+1)*d2, use_bias=False, kernel_initializer=k2,activation="elu",kernel_constraint=mask_1()),
        Dense(units = (latent_dim2+1)*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim2+1,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
        ])

    self.G_DEC = Sequential([
        InputLayer(shape=(latent_dim2+1,)),
        Dense(units = latent_dim2+1, use_bias=False,kernel_initializer='zeros',activation="linear",kernel_constraint=mask_0()),
        Dense(units = (latent_dim2+1)*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_1()),
        Dense(units = (latent_dim2+1)*d2, use_bias=False,kernel_initializer=k2,activation="elu",kernel_constraint=mask_2()),
        Dense(units=latent_dim2+1,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
        ])





  #@tf.function

  def enc(self, x): #returns encoded noise eps
    ec = self.Enc(x,training=True)
    mean, lv = tf.split(ec,num_or_size_splits=2,axis=1)
    return mean, lv #lv: log-variance


  def reparam(self, mean, lv):
    eps = tf.random.normal(shape=mean.shape) #seed not fixed for randomness
    return eps*tf.math.exp(lv*0.5) + mean #lv: log variance

  def lat_split(self,x):
    lat1, lat2 = tf.split(x, [self.latent_dim1, self.latent_dim2],axis=1)
    return lat1, lat2

  def dec(self,z, sigmoid=False):
    result = self.Dec(z,training=True)
    if sigmoid == True:
        result = tf.math.sigmoid(result)
        return result
    return result


  mse_loss = tf.keras.losses.MeanSquaredError()
  def gae_loss(self,x,y): #y used for weak supervision
    mean, lv = self.enc(x)
    z = self.reparam(mean, lv)
    z1,z2 = self.lat_split(z)
    z2 = tf.concat([z2,y.reshape(-1,1)],axis=1)
    z2_ = self.G_ENC(z2,training=True)
    z2_hat = self.G_DEC(z2_,training=True)
    return mse_loss(z2,z2_hat)



  b_loss2 = tf.keras.losses.BinaryCrossentropy()
  def C_ELBO_loss(model, x): #ELBO loss
    mean, lv = model.enc(x)
    z = model.reparam(mean, lv)
    z1,z2 = model.lat_split(z)
    z_ = tf.concat([z1,z2],axis=1)
    x_hat = model.dec(z_,sigmoid=False)
    flat = tf.keras.layers.Flatten()
    f_x_hat = flat(x_hat)
    f_x = flat(x)
    loss_2 = tf.norm(z, ord=2)
    return b_loss2(f_x,f_x_hat)+0.001*loss_2/2 #weighted, 0.001

  def rbf_(self,x,sig):
    x_norm = tf.reduce_sum(tf.square(x),axis=1,keepdims=True)
    dist = x_norm-2*tf.matmul(x,x,transpose_b=True)+tf.transpose(x_norm)
    return tf.exp(-dist/(2*sig**2))

  def HSIC(self,x,y,sig):
    x = tf.cast(x,tf.float32)
    y = tf.cast(y,tf.float32)
    if len(y.shape) == 1:
        y = tf.expand_dims(y, axis=1)
    n = tf.shape(x)[0]
    K = self.rbf_(x,sig)
    L = self.rbf_(y,sig)
    n_f = tf.cast(n,tf.float32)
    H = tf.eye(n, dtype=tf.float32)-tf.ones((n,n),dtype=tf.float32)/(n_f+1e-5)
    # centered kernels
    Kc = tf.matmul(tf.matmul(H, K), H)
    Lc = tf.matmul(tf.matmul(H, L), H)

    #normalized HSIC
    num = tf.linalg.trace(tf.matmul(Kc, Lc))
    den = tf.sqrt(tf.linalg.trace(tf.matmul(Kc, Kc)) * tf.linalg.trace(tf.matmul(Lc, Lc))) + 1e-5
    hsic_norm = num / den

    return hsic_norm

  def MI_loss(model,x,y,b):
    mean,lv = model.enc(x)
    z = model.reparam(mean,lv)
    z1, z2 = model.lat_split(z)
    hs_z1 = model.HSIC(z1,y,5e-3)
    hs_z2 = model.HSIC(z2,y,5e-3)
    MI_total = hs_z1-hs_z2*b
    return MI_total

  def MI_loss2(model,x,y,b):
    mean,lv = model.enc(x)
    z = model.reparam(mean,lv)
    z1, z2 = model.lat_split(z)
    hs_z1 = 0; hs_z2 = 0
    for i in range(z1.shape[1]):
      hs_z1 += model.HSIC(tf.expand_dims(z1[:,i], axis=1),y,5e-3)
    for j in range(z2.shape[1]):
      hs_z2 += model.HSIC(tf.expand_dims(z2[:,j], axis=1),y,5e-3)
    hs_z1 = hs_z1/z1.shape[1]
    hs_z2 = hs_z2/z2.shape[1]
    MI_total = hs_z1-hs_z2*b
    return MI_total

  #redundancy loss
  def MI_loss3(model,x,y):
    mean,lv = model.enc(x)
    z = model.reparam(mean,lv)
    z1, z2 = model.lat_split(z)
    hs_z12 = 0
    for j in range(z2.shape[1]-1):
      for i in range(j+1,z2.shape[1],1):
        hs_z12 += model.HSIC(tf.expand_dims(z1[:,j], axis=1),z1[:,i],5e-3)
    hs_z12 = hs_z12/(z2.shape[1]*(z2.shape[1]-1)/2)
    MI_total = hs_z12
    return MI_total

################Defining Model

#Updating LightHCG model

tf.keras.utils.set_random_seed(987)
os.environ['TF_DETERMINISTIC_OPS']='1'

tf.executing_eagerly()
import tensorflow.keras.backend as K

Epochs = 400
lat_dim_1 = 4
lat_dim_2 = 3
LightHCG = lighthcg(224,lat_dim_1,lat_dim_2,16,32,16,16,128,16,4)
mse_loss = tf.keras.losses.MeanSquaredError()
b_loss2 = tf.keras.losses.BinaryCrossentropy()


alpha=0.6 #
i = 0
rho = 0.1 #
gamma=0.9
beta = 1.01
#lamb = 1.0 #L1-regularization


loss_of_cs = []
loss_of_cv = []
basic_opt0 = tf.keras.optimizers.Adam(learning_rate=0.005)
basic_opt = tf.keras.optimizers.Adam(learning_rate=0.002) #
basic_opt2 = tf.keras.optimizers.Adam(learning_rate=0.002)
basic_opt3 = tf.keras.optimizers.Adam(learning_rate=0.0005) #
basic_opt4 = tf.keras.optimizers.Adam(learning_rate=0.0005) #


while i < Epochs:
    with tf.GradientTape() as dv_t0, tf.GradientTape() as dv_t, tf.GradientTape() as dv_t2, tf.GradientTape() as dv_t3, tf.GradientTape() as dv_t4:
      loss_ = LightHCG.C_ELBO_loss(X_tr)
      loss_2 = LightHCG.gae_loss(X_tr,tag_tr)
      loss_3 = LightHCG.MI_loss2(X_tr,tag_tr,1.5) #1.5

      h_a = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(LightHCG.G_DEC.weights[0], LightHCG.G_DEC.weights[0])))-lat_dim_2-1 #
      if i < 50:
        cs_l = loss_ + loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2
      elif i < 100:
        cs_l = loss_ + loss_3*5 + loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2
      else:
        loss_4 = LightHCG.MI_loss3(X_tr,tag_tr)
        cs_l = loss_*2.0 + loss_3*5 + loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2+loss_4*0.5
      d_l = loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2

    loss_of_cs.append(cs_l)
    loss_of_cv.append(d_l)
    grad_g_0 = dv_t0.gradient(d_l, [LightHCG.G_DEC.trainable_variables[0]])
    grad_g_a = dv_t.gradient(d_l, LightHCG.G_DEC.trainable_variables[1:])
    grad_g_b = dv_t2.gradient(d_l, LightHCG.G_ENC.trainable_variables)
    grad_g_c = dv_t3.gradient(cs_l, LightHCG.Enc.trainable_variables)
    grad_g_d = dv_t4.gradient(cs_l, LightHCG.Dec.trainable_variables) #cs_l
    basic_opt0.apply_gradients(zip(grad_g_0, [LightHCG.G_DEC.trainable_variables[0]]))
    basic_opt.apply_gradients(zip(grad_g_a, LightHCG.G_DEC.trainable_variables[1:]))
    basic_opt2.apply_gradients(zip(grad_g_b, LightHCG.G_ENC.trainable_variables))
    basic_opt4.apply_gradients(zip(grad_g_c,LightHCG.Enc.trainable_variables))
    basic_opt3.apply_gradients(zip(grad_g_d, LightHCG.Dec.trainable_variables))
    h_a_new = tf.linalg.trace(tf.linalg.expm(tf.math.multiply(LightHCG.G_DEC.weights[0], LightHCG.G_DEC.weights[0])))-lat_dim_2-1
    alpha =  alpha + rho * h_a_new
    if (tf.math.abs(h_a_new) >= gamma*tf.math.abs(h_a)):
        rho = beta*rho
    else:
        rho = rho
    if (i+1) %10 == 0: print(i+1, cs_l,d_l,loss_,loss_3)
    i = i+1

###########################

#causal weighted adjacency matrix(fitted)
import seaborn as sns
sns.set_style("darkgrid")
sns.heatmap(np.array(LightHCG.G_DEC.weights[0]), cmap="vlag",center=0)
plt.show()

#binarized adjacency matrix
import seaborn as sns
sns.set_style("darkgrid")
sns.heatmap(np.where(np.abs(np.array(LightHCG.G_DEC.weights[0]))> np.quantile(np.abs(np.array(LightHCG.G_DEC.weights[0])),0.75),1,0), cmap="gray_r",linewidths=1,linecolor="black")
plt.show()

tf.keras.utils.set_random_seed(456)
mean, lv = LightHCG.enc(X_tr)
z = LightHCG.reparam(mean, lv)
z1,z2 = LightHCG.lat_split(z)
z_ = tf.concat([z1,z2],axis=1)
x_hat = LightHCG.dec(z_, sigmoid=False)

#Mutual Information for each latent variable in Z
print(mutual_info_regression(np.array(z1[:,0]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z1[:,1]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z1[:,2]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z1[:,3]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z2[:,0]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z2[:,1]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))
print(mutual_info_regression(np.array(z2[:,2]).reshape(-1,1),tag_tr,n_neighbors=5,random_state=321))

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cv, color="black",label="Loss_2")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cs, color="red",label="Loss_1")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

d_l #0.0705247670412063

loss_of_cs[-1] #-3.6897904872894287

print(loss_,loss_2,loss_3)


from sklearn.preprocessing import StandardScaler, MinMaxScaler

st1 = StandardScaler()

z2_tr = st1.fit_transform(z2)

#TSNE results for LightHCG fitting

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=20, learning_rate=100, random_state=123)

X_tsne = tsne.fit_transform(z2_tr)

tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['Target'] = tag_tr

sns.scatterplot(x='TSNE1',y='TSNE2',hue='Target',palette="hls",data=tsne_df,legend='full',alpha=0.9)
plt.title('t-SNE Visualization of Z2 space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

st2 = StandardScaler()

z1_tr = st2.fit_transform(z1)

tsne1 = TSNE(n_components=2, perplexity=20, learning_rate=100, random_state=123)

X_tsne1 = tsne1.fit_transform(z1_tr)

tsne_df1 = pd.DataFrame(data=X_tsne1, columns=['TSNE1', 'TSNE2'])
tsne_df1['Target'] = tag_tr

sns.scatterplot(x='TSNE1',y='TSNE2',hue='Target',palette="hls",data=tsne_df1,legend='full',alpha=0.9)
plt.title('t-SNE Visualization of Z1 space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')



####################Qualitative Checks for disentanglement

tf.keras.utils.set_random_seed(321321)
mean, lv = LightHCG.enc(X_tr)
z = LightHCG.reparam(mean, lv)
z1,z2 = LightHCG.lat_split(z)
z_ = tf.concat([z1,z2],axis=1)

print(np.min(z_[:,3]),np.max(z_[:,3]))
print(np.min(z_[:,4]),np.max(z_[:,4]))
print(np.min(z_[:,5]),np.max(z_[:,5]))

#Z2,0
set1 = [0,1,2,3,4,5]
fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=6
ax3 = []
diff2 = []
mx =0
mn =1

for m in range(6):
  np.random.seed(321)
  diff1 = 0
  for i in range(50):
    ix = np.random.randint(0,352,1)[0]
    distangle = np.array(z_[ix,:])
    distangle[4]  = -2
    distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,7])
    distg = LightHCG.dec(distg, sigmoid=False)

    distangle2 = np.array(z_[ix,:])
    distangle2[4] =  set1[m]
    distg2 = tf.reshape(tf.convert_to_tensor(distangle2, dtype=tf.float32),[1,7])
    distg2 = LightHCG.dec(distg2, sigmoid=False)

    diff1 += np.abs(distg[0,:,:,:]-distg2[0,:,:,:])
  diff1 = diff1/50
  diff2.append(diff1)
  if mx < np.max(diff1):
    mx = np.max(diff1)
  if mn > np.min(diff1):
    mn = np.min(diff1)


for m in range(6):
  dff = diff2[m]
  dff = (dff-mn)/(mx-mn)
  dff = np.where(dff>np.quantile(dff,0.75),dff,0.0)
  ax3.append(fig3.add_subplot(rows,columns,m+1))
  ax3[-1].set_title("Z2,0:"+str(set1[m]))
  plt.axis("off")
  plt.imshow(dff)

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Z2,2
set1 = [-1.4,-1.0,-0.6,-0.2,0.2,0.6]
fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=6
ax3 = []
diff2 = []
mx =0
mn =1

for m in range(6):
  np.random.seed(4321)
  diff1 = 0
  for i in range(50):
    ix = np.random.randint(0,352,1)[0]
    distangle = np.array(z_[ix,:])
    distangle[6]  = 0.8
    distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,7])
    distg = LightHCG.dec(distg, sigmoid=False)

    distangle2 = np.array(z_[ix,:])
    distangle2[6] =  set1[5-m]
    distg2 = tf.reshape(tf.convert_to_tensor(distangle2, dtype=tf.float32),[1,7])
    distg2 = LightHCG.dec(distg2, sigmoid=False)

    diff1 += np.abs(distg[0,:,:,:]-distg2[0,:,:,:])
  diff1 = diff1/50
  diff2.append(diff1)
  if mx < np.max(diff1):
    mx = np.max(diff1)
  if mn > np.min(diff1):
    mn = np.min(diff1)


for m in range(6):
  dff = diff2[m]
  dff = (dff-mn)/(mx-mn)
  dff = np.where(dff>np.quantile(dff,0.75),dff,0.0)
  ax3.append(fig3.add_subplot(rows,columns,m+1))
  ax3[-1].set_title("Z2,2:"+str(set1[5-m]))
  plt.axis("off")
  plt.imshow(dff)

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Z2,1
set1 = [-0.06,-0.05,-0.04,-0.03,-0.02,-0.01]
fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=6
ax3 = []
diff2 = []
mx =0
mn =1

for m in range(6):
  np.random.seed(4321)
  diff1 = 0
  for i in range(50):
    ix = np.random.randint(0,352,1)[0]
    distangle = np.array(z_[ix,:])
    distangle[5]  = -0.07
    distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,7])
    distg = LightHCG.dec(distg, sigmoid=False)

    distangle2 = np.array(z_[ix,:])
    distangle2[5] =  set1[m]
    distg2 = tf.reshape(tf.convert_to_tensor(distangle2, dtype=tf.float32),[1,7])
    distg2 = LightHCG.dec(distg2, sigmoid=False)

    diff1 += np.abs(distg[0,:,:,:]-distg2[0,:,:,:])
  diff1 = diff1/50
  diff2.append(diff1)
  if mx < np.max(diff1):
    mx = np.max(diff1)
  if mn > np.min(diff1):
    mn = np.min(diff1)


for m in range(6):
  dff = diff2[m]
  dff = (dff-mn)/(mx-mn)
  dff = np.where(dff>np.quantile(dff,0.75),dff,0.0)
  ax3.append(fig3.add_subplot(rows,columns,m+1))
  ax3[-1].set_title("Z2,1"+str(set1[m]))
  plt.axis("off")
  plt.imshow(dff)

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set1 = [-3,-2,-1,0,1,2,3] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

for i in range(7):
  distangle = np.array(z_[50,:])
  distangle[4] =  set1[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,7])
  distg = LightHCG.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z2,0:"+str(set1[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,1])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set1 = [-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5]  #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

for i in range(7):
  distangle = np.array(z_[50,:])
  distangle[6] = set1[6-i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,7])
  distg = LightHCG.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z2,2:"+str(set1[6-i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,1])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

#Disentanglement simulation (Qualitative)
set1 = [-2.1,-1.8,-1.5,-1.2,-0.9,-0.6,-0.3] #

fig3 = plt.figure(figsize=(15,2))
rows = 1; columns=7
ax3 = []

for i in range(7):
  distangle = np.array(z_[50,:])
  distangle[5] = set1[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,7])
  distg = LightHCG.dec(distg, sigmoid=False)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z2,1:"+str(set1[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,1])

#Visualized Results
plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.5, wspace = 0)
plt.show()

np.random.seed(321)
diff1 = 0
for i in range(50):
  ix = np.random.randint(0,352,1)[0]
  distangle = np.array(z_[ix,:])
  distangle[4]  = -1
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,7])
  distg = LightHCG.dec(distg, sigmoid=False)

  distangle2 = np.array(z_[ix,:])
  distangle2[4] =  1
  distg2 = tf.reshape(tf.convert_to_tensor(distangle2, dtype=tf.float32),[1,7])
  distg2 = LightHCG.dec(distg2, sigmoid=False)

  diff1 += np.abs(distg[0,:,:,:]-distg2[0,:,:,:])
#normalize difference
diff1 = diff1/50
diff1 = (diff1-np.min(diff1))/(np.max(diff1)-np.min(diff1))
diff1 = np.where(diff1>np.quantile(diff1,0.8),diff1,0.0)
plt.imshow(diff1)
plt.grid()

###################

plt.imshow(X_tr[50,:,:,:])#
print(tag_tr[50])
plt.grid()
plt.show()

#Downstream task experiments: encoding train/test data for analysis based on LightHCG encoder.
tf.keras.utils.set_random_seed(4321)
mean1, lv1 = LightHCG.enc(X_tr)
z_tr = LightHCG.reparam(mean1, lv1)
z_tr1,z_tr2 = LightHCG.lat_split(z_tr)
z_tr = tf.concat([z_tr1,z_tr2],axis=1)

tf.keras.utils.set_random_seed(4321)
mean2, lv2 = LightHCG.enc(X_te3)
z_te = LightHCG.reparam(mean2, lv2)
z_te1,z_te2 = LightHCG.lat_split(z_te)
z_te = tf.concat([z_te1,z_te2],axis=1)





#Only using latent space Z2 for downstream tasks.
z_tr_ = np.array(z_tr2)
z_te_ = np.array(z_te2)

LightHCG.Enc.summary()

#DNN for predicting Glaucoma
tf.keras.utils.set_random_seed(321321)
os.environ['TF_DETERMINISTIC_OPS']='1'
k3 = tf.keras.initializers.HeNormal(123)

md_c = Sequential([
    InputLayer(shape=(3,)),
    BatchNormalization(),
    Dense(units = 32, kernel_initializer=k3, activation="elu"), #
    BatchNormalization(),
    Dropout(0.05),
    Dense(units = 64, kernel_initializer=k3, activation="elu"),#
    BatchNormalization(),
    Dense(units = 32, kernel_initializer=k3, activation="elu"), #
    BatchNormalization(),
    Dense(units=1,kernel_initializer=k3, activation="sigmoid")
    ])

md_c.summary()

from tensorflow.keras.optimizers import RMSprop, Adam
md_c.compile(optimizer=Adam(learning_rate=1e-4), loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])

hist2 = md_c.fit(x=z_tr_, y=tag_tr,epochs=300,batch_size=100,shuffle=True) #500

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(hist2.history['loss'], color="red",label="loss")

f1_score(tag_tr,np.round(md_c.predict(z_tr_)))

accuracy_score(tag_tr, np.round(md_c.predict(z_tr_)))

#Prediction results
prd = md_c.predict(z_te_)
prd1 = []
for i in range(len(tag_te)):
  if prd[i] < 0.5:
    prd1.append(0)
  else:
    prd1.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(tag_te,prd1), display_labels=[0,1])
dis.plot()
plt.grid()
plt.show()

from sklearn.metrics import classification_report

print(classification_report(tag_te,prd1,digits=4))

from sklearn.metrics import precision_score, recall_score,f1_score
print(precision_score(tag_te,prd1))
print(recall_score(tag_te,prd1))
print(f1_score(tag_te,prd1))

from sklearn.metrics import roc_auc_score

roc_auc_score(tag_te,md_c.predict(z_te_))

