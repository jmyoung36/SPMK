# #!/bin/sh

# directory containing images to be masked
IMDIR=/home/jonyoung/IoP_data/Data/IXI/GM-masked-groupwise-sub1pt5

# get the number of files to be processed
n_images="`ls $IMDIR/*sub1pt5.nii.gz | wc -l`"
echo "$n_images image files to be processed"

i=0

# change $Case etc appropriately to match pattern of the type of file we are interested in.
for Case in `ls $IMDIR/*sub1pt5.nii.gz`
do
      let i++
      filename=`basename ${Case} .nii.gz`
      fslmaths ${IMDIR}/${filename} -s ${1} ${IMDIR}/${filename}_smooth_${1}.nii.gz
         
done


