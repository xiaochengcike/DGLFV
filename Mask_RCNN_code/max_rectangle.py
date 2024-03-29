#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:58:05 2018

@author: Ashiki
"""
import cv2
import time
import skimage
import numpy as np 
'''Give image path and run script'''

# computes rectangle area
def compute_area(ymax,ymin,xmax,xmin):
	return (ymax-ymin)*(xmax-xmin)
	
# define rectangle
coord_upper_left={"x":0,"y":0}
coord_lower_left={"x":0,"y":0}
coord_upper_right={"x":0,"y":0}
coord_lower_right={"x":0,"y":0}
xmax=0
xmin=0
ymax=0
ymin=0

#Booleans that not take into account zero pixels
upper_left=False
lower_left=False

unhindered=[]

# reset variables 

def reset_variables():
	global	coord_upper_left
	global	coord_lower_left
	global	coord_upper_right
	global	coord_lower_right
	global	upper_left
	global	lower_left
	global	unhindered
	
	coord_upper_left={"x":0,"y":0}
	coord_lower_left={"x":0,"y":0}
	coord_upper_right={"x":0,"y":0}
	coord_lower_right={"x":0,"y":0}
	upper_left=False
	lower_left=False
	unhindered=[]

def find_max_rectangle(image_path):
	start=time.time()
	global	coord_upper_left
	global	coord_lower_left
	global	coord_upper_right
	global	coord_lower_right
	global	upper_left
	global	lower_left
	global	unhindered
	global  real_image
	#Initial area of our first rectangle
	area=0

	#get image, binarize it if necessary
	real_image=cv2.imread(image_path)
	image=cv2.imread(image_path,0)
	ret,bin_image = cv2.threshold(image,128,1,cv2.THRESH_BINARY)
		
	#Shape of image
	height, width = bin_image.shape
	for i in range(height):

		#Reset all variables
		reset_variables()	
		#Dealing with the case of last row 

		#Check that our current area is larger than the max of area remaining to check
		#If so no need to continue
		if(area>(width*(height-i))):
			break
				
		for j in range(width):
			#Find the first line of pixels containing "1"
			if(bin_image[i,j]==0 and not(upper_left)):
				# Do not consider pixels that are equal to 0 unless upper left bound is ddefined
				continue
			if(bin_image[i,j]==1 and not(upper_left)):
				#We found our "1" pixel that defines our upper left coordinate
				coord_upper_left["x"]=j
				coord_upper_left["y"]=i
				if(j==(width-1)):
					coord_upper_right["y"]=i
					coord_upper_right["x"]=j
				upper_left=True
			#define our upper right coordinate after upper left coordinate has been set
			if((bin_image[i,j]==0 and upper_left) or (bin_image[i,j]==1 and j==(width-1) and upper_left) ):
				coord_upper_right["x"]=j-1
				coord_upper_right["y"]=i
				if(j==(width-1)):
					coord_upper_right["x"]=j
				upper_left=False
				
				#Vertical evaluation of previously found line through rows
				#Horizontal and vertical counters for evaluation
				for horizontal_counter in range(coord_upper_left["x"],(coord_upper_right["x"]+1)):
					for vertical_counter in range((i+1),height):
						#iteratively check rectangles using lower left tracker
						#we hit a bound when we meet a '0' pixel or we hit the height
						if(bin_image[vertical_counter,horizontal_counter]==0 and not(lower_left)):
							lower_left=True
							coord_lower_left["x"]=horizontal_counter
							coord_lower_left["y"]=vertical_counter-1
							#compute the area for this particular case
							a=vertical_counter-coord_upper_left["y"]
							#check to see if a larger area exists
							#if so set rectangle coordinates
							if(a>area):
								area=a
								ymax=height-coord_upper_left["y"]
								ymin=height-coord_lower_left["y"]-1
								xmax=coord_lower_left["x"]+1
								xmin=coord_lower_left["x"]
							#No need to continue downward, we have our first vertical line
							#so we break the vertical counter loop
							break
						#if we hit the bottom and we find no lower left bound
						#we set a lower left coordinate to last element of the vertical line
						if(vertical_counter==height-1 and bin_image[vertical_counter,horizontal_counter]==1 and not(lower_left)):
							lower_left=True
							coord_lower_left["x"]=horizontal_counter
							coord_lower_left["y"]=vertical_counter
							#compute area and compare it 
							a=height-coord_upper_left["y"]
							if(a>area):
								area=a
								ymax=height-coord_upper_left["y"]
								ymin=height-coord_lower_left["y"]-1
								xmax=coord_lower_left["x"]+1
								xmin=coord_lower_left["x"]
							break
						#lower left coordinate has already been set
						#so we are basically checking vertical lines along our initial pixel line at the top
						if((bin_image[vertical_counter,horizontal_counter]==0 and lower_left)):
							if(coord_lower_left["y"]<vertical_counter-1):
							
								len_unhindered=len(unhindered)
								#we do not want to make hindered rectangles unhindered so we have to check that they are not already set
								already=False
								for l in range(len_unhindered):
									if(unhindered[l][0]==coord_lower_left["y"]):
										already=True
								if(not(already)):		
									unhindered.append([coord_lower_left["y"],coord_lower_left["x"],coord_upper_left["y"],coord_upper_left["x"]])
								#we set new lower bounds and upper bounds accordingly
								coord_lower_left["x"]=horizontal_counter
								coord_lower_left["y"]=vertical_counter-1
								coord_upper_left["x"]=horizontal_counter
								#upper counter "y" coordinate remains the same
								#compute the area and compare it accordignly
								a=vertical_counter-coord_upper_left["y"]
								if(a>area):
									area=a
									ymax=height-coord_upper_left["y"]
									ymin=height-coord_lower_left["y"]-1
									xmax=horizontal_counter+1
									xmin=coord_upper_left["x"]
								#Now we compute areas of above unhindered rectangles
								#and compare their areas
								length_unhindered=len(unhindered)
								
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]
								break
							if((coord_lower_left["y"]>(vertical_counter-1))):
								#first insert a new unhindered element lower bound between current hindered elements
								#correct particular exceptions
								length_unhindered=len(unhindered)
								checked=False
								added=False
								for l in range(length_unhindered):
									if(unhindered[l][0]<vertical_counter-1):
										checked=True
									if(unhindered[l][0]>vertical_counter-1):
										unhindered.insert(l,[vertical_counter-1,unhindered[l][1],unhindered[l][2],unhindered[l][3]])
										added=True
										break
								if(checked and not(added)):
									unhindered.append([(vertical_counter-1),(coord_lower_left["x"]),coord_upper_left["y"],coord_lower_left["x"]])
								#now get rid of hindered elements
								length_unhindered=len(unhindered)
								if(length_unhindered!=0):
									indices=[]
									for l in range(length_unhindered):
										if(unhindered[l][0]>(vertical_counter-1)):
											indices.append(l)
									indices.reverse()
									for indice in indices:
										unhindered.pop(indice)
								#compute remaining areas
								#first check to see if there were indeed unhidered elements previously created
								#compute and compare their areas
								if(length_unhindered!=0):
									length_unhindered=len(unhindered)
									coord_lower_left["y"]=vertical_counter-1
									coord_lower_left["x"]=unhindered[length_unhindered-1][1]+1
									for l in range(length_unhindered):
										unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
										if(unhindered_area>area):
											area=unhindered_area
											ymax=(height-unhindered[l][2])
											ymin=(height-unhindered[l][0]-1)
											xmax=(horizontal_counter+1)
											xmin=unhindered[l][1]
										
								if(length_unhindered==0):
									coord_lower_left["y"]=vertical_counter-1
									#compute one area
									a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
									if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]
								break
							#if we stay at the same lower bound
							if((coord_lower_left["y"]==(vertical_counter-1))or(coord_lower_left["y"]==vertical_counter and vertical_counter==height-1)):
								#compute and compare
								a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
								if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]
								#check unhindered elements
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]
								break
							break
						#Special case where we hit the bottom
						if((bin_image[vertical_counter,horizontal_counter]==1 and lower_left and vertical_counter==(height-1))):
							if(coord_lower_left["y"]<vertical_counter):
								len_unhindered=len(unhindered)
								already=False
								for l in range(len_unhindered):
									if(unhindered[l][0]==coord_lower_left["y"]):
										already=True
								if(not(already)):		
									unhindered.append([coord_lower_left["y"],coord_lower_left["x"],coord_upper_left["y"],coord_upper_left["x"]])
								#we set new lower bounds and upper bounds accordingly
								coord_lower_left["x"]=horizontal_counter
								coord_lower_left["y"]=vertical_counter
								coord_upper_left["x"]=horizontal_counter
								#upper counter "y" coordinate remains the same
								#compute the area and compare it accordignly
								a=height-coord_upper_left["y"]
								if(a>area):
									area=a
									ymax=height-coord_upper_left["y"]
									ymin=height-coord_lower_left["y"]-1
									xmax=horizontal_counter+1
									xmin=coord_upper_left["x"]
								#Now we compute areas of above unhindered rectangles
								#and compare their areas
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]
								break
							if((coord_lower_left["y"]>(vertical_counter))):
								#first insert a new unhindered element lower bound between current hindered elements
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									if(unhindered[l][0]>vertical_counter-1):
										unhindered.insert(l,[vertical_counter-1,unhindered[l][1],unhindered[l][2],unhindered[l][3]])
										break
								#now get rid of hindered elements
								length_unhindered=len(unhindered)
								if(length_unhindered!=0):
									indices=[]
									for l in range(length_unhindered):
										if(unhindered[l][0]>(vertical_counter-1)):
											indices.append(l)
									indices.reverse()
									for indice in indices:	
										unhindered.pop(indice)
								#compute remaining areas
								#compute and compare their areas
								if(length_unhindered!=0):
									length_unhindered=len(unhindered)
									coord_lower_left["y"]=vertical_counter-1
									coord_lower_left["x"]=unhindered[length_unhindered-1][1]+1
									for l in range(length_unhindered):
										unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
										if(unhindered_area>area):
											area=unhindered_area
											ymax=(height-unhindered[l][2])
											ymin=(height-unhindered[l][0]-1)
											xmax=(horizontal_counter+1)
											xmin=unhindered[l][1]
										
								if(length_unhindered==0):
									coord_lower_left["y"]=vertical_counter-1
									#compute one area
									a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))
									if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]
								break
							#if we stay at the same lower bound
							if((coord_lower_left["y"]==(vertical_counter-1))or(coord_lower_left["y"]==vertical_counter and vertical_counter==height-1)):
								#compute and compare
								a=compute_area((height-coord_upper_left["y"]),(height-coord_lower_left["y"]-1),(horizontal_counter+1),(coord_lower_left["x"]))

								if(a>area):
										area=a
										ymax=height-coord_upper_left["y"]
										ymin=height-coord_lower_left["y"]-1
										xmax=horizontal_counter+1
										xmin=coord_lower_left["x"]								
								length_unhindered=len(unhindered)
								for l in range(length_unhindered):
									unhindered_area=compute_area((height-unhindered[l][2]),(height-unhindered[l][0]-1),(horizontal_counter+1),unhindered[l][1])
									if(unhindered_area>area):
										area=unhindered_area
										ymax=(height-unhindered[l][2])
										ymin=(height-unhindered[l][0]-1)
										xmax=(horizontal_counter+1)
										xmin=unhindered[l][1]				
								break
							break
				reset_variables()
	end=time.time()
#	print("the process took %lf seconds" %(end-start))
	cv2.rectangle(real_image, (xmin,height-ymax), (xmax-1,height-ymin-1),(255,0,0), thickness=1, lineType=8, shift=0)
	results = [xmin,(height-ymax),(xmax-1),(height-ymin-1)]
	return results
	
if __name__ == "__main__":	
    image_path="./save_result/mask_023.png"	
    a=find_max_rectangle(image_path)
    print(a)
#    cv2.imshow("image",real_image)
#    cv2.waitKey(0)
    skimage.io.imshow(real_image)
#    skimage.io.imsave("./save_result/reg_007.png",real_image)
    s = (a[2]-a[0])*(a[3]-a[1])
    all_s = np.sum(real_image[:,:,0])/255
    s_percent = s/all_s
    print(s_percent)
    
    