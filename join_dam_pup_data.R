#Hannah Lapp
#Last updated: April  4, 2023

#Join deeplabcut output from dams and pups into one file
#Pup dlc output are the raw detections (before individual assembly), after the pickle file has been converted
#to csv format using the PhenoPickleRaw.py script

#Load libraries ------
library(stringr)
library(tidyverse)


#Set working directory-----
#working directory should contain sub directories for: 
#1) dam dlc tracking
#2) pup dlc detection csvs
#3) the joined dam and pup data files generated with this script

#replace the path below
setwd("your/directory/here")

#replace the file path below with the path to your pup dlc data:
pup_path <- "./pup_dlc_dections/path/"

#replace the file path below with the path to your dam dlc data:
dam_path <- "./dam_dlc_csvs/path/"

#replace the file path below with the path to the output subdirectory,
#where the joined files will be written:
output_path <- "./dam_pup_points_joined/"


#Create lists of dlc data files --------
#create a list of pup files from the pup dlc raw detections unpickled subdirectory
pup_files <- list.files(pup_path, pattern = ".csv")

#create a list of dam files from the dam dlc subdirectory
dam_files <- list.files(dam_path, pattern = ".csv")


#Create keys------
#This will create "keys" to match files names from dam and pup files to join the 
#correct videos together. Default full file names will not match because dlc 
#automatically appends the model name to the dlc tracking output. 
#In the code below, I have taken the characters 1-7 from the file names because 
#they are unique names in my data set.
#If your data set requires more (or fewer) characters to ensure pup and dam file 
#names match, please adjust accordingly 

dam_keys <- substr(dam_files, 1, 7)  
pup_keys <- substr(pup_files, 1, 7) 

#Now, a "master key" containing all of the file names that match between the dam 
#and pup lists is created
master_key <- intersect(dam_keys, pup_keys)

#Alternatively, you can specify a list of specific videos to join the files:
#master_key  <- c("F18_10", "F2_10_", "F2_12_", "F5_06_")

#This line creates the start of the name of the output file containing joined data
out_name_list <- str_replace(dam_files, ".csv", "") 

#Generate your joined files-----
#Run the loop below to generate joined dam_pup tracking files for all the files 
#on the master list. Files will be written to the output directory you specified.

for(i in 1:length(master_key)){
  d <- as.numeric(match(1,str_detect(master_key[i], dam_keys))) #find location of master key string in dam_key (index)
  dam <- read.csv(paste(dam-path, dam_files[d], sep = ""))
  
  p <- as.numeric(match(1,str_detect(master_key[i], pup_keys))) #find location of master key string in pup_key (index)
  pup <- read.csv(paste(pup_path, pup_files[p], sep = ""))
  
  pup <- pup[,1:325]
  
  pup[,] <- lapply(pup, as.numeric)

  colnames(dam) <- paste(dam[1,], dam[2,])
  colnames(dam)[1] <- "frame"
  dam <- dam[-c(1,2),]
  dam[,] <- lapply(dam, as.numeric)
  
  all <- left_join(dam, pup, by = "frame")
  
  all_pt <- as.data.frame(matrix(nrow= (nrow(all)+3), ncol = ncol(all)))
  
  all_pt[4:nrow(all_pt),1] <- all[,1] #assign row numbers
  
  all_pt[1,1] <- "scorer"
  all_pt[2,1] <- "bodyparts"
  all_pt[3,1] <- "coords"
  all_pt[1,2:ncol(all)] <- "DLC_resnet50_dam_and_pup"
  
  bodyparts <- colnames(all) %>% 
    str_replace(., "_x", "") %>% 
    str_replace(., " x", "") %>% 
    str_replace(., "_y", "") %>% 
    str_replace(., " y", "") %>% 
    str_replace(., "_likelihood", "") %>% 
    str_replace(., " likelihood", "") 
  
  all_pt[2,2:ncol(all)] <- bodyparts[2:439]
  
  all_pt[3,2:ncol(all)] <- rep(c("x", "y", "likelihood"),times=(ncol(all)-1))
  
  all_pt[4:nrow(all_pt), ] <- all
  
  fwrite(all_pt, paste(output_path, out_name_list[d], 
                       ".csv", sep = ""), row.names = FALSE, col.names = FALSE)
}






