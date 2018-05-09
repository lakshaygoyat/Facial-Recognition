import cv2,os,numpy as np,time
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

#path where the data set is located
path = 'att_faces'; os.chdir(path)


def reduce_dimension(image_vector,reduce_matrix):
	return image_vector.dot(reduce_matrix.T)
	
def reduce_size(image,dim):
	return cv2.resize(img,dim)

# the image to be classified
sample_image = cv2.imread(os.path.join('s4','1.pgm'),0)
#cv2.imshow('as',sample_image)

sampleimagedimensions = cv2.imread(os.path.join('s1','1.pgm'),0).shape; temp = 110.0
r = temp / sampleimagedimensions[1]; dim = (int(temp),int(sampleimagedimensions[0] * r))
imagefiledirectories,raw_images,labels = [directory for directory in os.listdir(os.getcwd())],[],[]

for imagedirectory in imagefiledirectories:
    if imagedirectory.startswith('s'):
        for imgfile in os.listdir(imagedirectory):
            if imgfile.endswith('.pgm'):
                raw_images.append(cv2.resize(cv2.imread(imagedirectory+'/'+imgfile,0),dim))
                labels.append(int(imagedirectory[1:]))
				
feature_matrix,labels = np.array([raw_image.ravel() for raw_image in raw_images]),np.array(labels)
pca = PCA(n_components=20); pca.fit(feature_matrix); eigen_faces = pca.components_
norm_eigen_faces = [cv2.normalize(eigen_faces[i].reshape(-1,dim[0]),0,255,cv2.NORM_MINMAX) for i in range(len(eigen_faces))]
eigenfacesimage = np.hstack((norm_eigen_faces[0],norm_eigen_faces[1]))

for eigen_face in norm_eigen_faces[2:]: 
    eigenfacesimage = np.hstack((eigenfacesimage,eigen_face))
	
cv2.imwrite('Eigenfaces.jpg',eigenfacesimage)
#print eigenfacesimage
cv2.imshow('eigenfaces',eigenfacesimage)
k = cv2.waitKey(0)
if k == ord('q') & 0xFF: cv2.destroyAllWindows()     

   
rfeature_matrix = feature_matrix.dot(eigen_faces.T)

classifier = LogisticRegression(); classifier.fit(rfeature_matrix,labels)
testimage = cv2.resize(sample_image,dim)
testimage = reduce_dimension(testimage.ravel().reshape(1,-1),eigen_faces)
print 'Subject: ',classifier.predict(testimage)
