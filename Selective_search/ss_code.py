import cv2
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET

def selective_search(img, strategy):
    """
    @brief Selective search with different strategies
    @param img The input image
    @param strategy The strategy selected ['color', 'all']
    @retval bboxes Bounding boxes
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()
    ##################################################
    # TODO: For this part, please set the K as 200,  #
    #       sigma as 0.8 for the graph segmentation. #
    #       Use gs as the graph segmentation for ss  #
    #       to process after strategies are set.     #
    ##################################################
    ss.addImage(img) # add image to ss
    gs.setSigma(0.8) #set sigma value
    gs.setK(200) #set k value
    ss.addGraphSegmentation(gs) #add graph parameters to graph
    if strategy == "color":
        c = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor() #select strategy
        ss.addStrategy(c) # add strategy to ss
    else:
        allSt = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor(),
                                                                                                    cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture(),
                                                                                                    cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize(),
                                                                                                    cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill())
        # add multiple strategy 
        ss.addStrategy(allSt)
    ##################################################
    # End of TODO                                    #
    ##################################################
    bboxes = ss.process() #get proposal based on the strategy
    xyxy_bboxes = []

    for box in bboxes:
        x, y, w, h = box
        xyxy_bboxes.append([x, y, x+w, y + h])

    return xyxy_bboxes

def parse_annotation(anno_path):
    """
    @brief Parse annotation files for ground truth bounding boxes
    @param anno_path Path to the file
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    gt_bboxes = []
    for child in root:
        if child.tag == 'object':
            for grandchild in child:
                if grandchild.tag == "bndbox":
                    x0 = int(grandchild.find('xmin').text)
                    x1 = int(grandchild.find('xmax').text)
                    y0 = int(grandchild.find('ymin').text)
                    y1 = int(grandchild.find('ymax').text)
                    gt_bboxes.append([x0, y0, x1, y1])
    return gt_bboxes

def bb_intersection_over_union(boxA, boxB):
    """
    @brief compute the intersaction over union (IoU) of two given bounding boxes
    @param boxA numpy array (x_min, y_min, x_max, y_max)
    @param boxB numpy array (x_min, y_min, x_max, y_max)
    """
    ##################################################
    # TODO: Implement the IoU function               #
    ##################################################
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) # calculate overlap
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea) #get iou value
    ##################################################
    # End of TODO                                    #
    ##################################################
    return iou

def visualize(img, boxes, color,name):
    """
    @breif Visualize boxes
    @param img The target image
    @param boxes The box list
    @param color The color
    """
    for box in boxes:
        ##################################################
        # TODO: plot the rectangles with given color in  #
        #       the img for each box.                    #
        ##################################################
        start = (box[0],box[1])
        end   = (box[2],box[3])
        image = cv2.rectangle(img, start, end, color,thickness = 2) # get the rectangle on the image
    #cv2.imshow(name, image)
    #save = name +".jpg"
    #cv2.imwrite(save, image)
        ##################################################
        # End of TODO                                   #
        ##################################################
    return image



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='color')

    args =parser.parse_args()
    img_dir = './HW2_Data/JPEGImages'
    anno_dir = './HW2_Data/Annotations'
    thres = .5

    

    img_list = os.listdir(img_dir)
    num_hit = 0
    num_gt = 0
    imgno = 1

    for img_path in img_list:
        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)
        ##################################################
        # TODO: Load the image with OpenCV               #
        ##################################################
        img = cv2.imread(img_name,1) # import image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert imported BGR image to RGB

        ##################################################
        # End of TODO                                    #
        ##################################################

        proposals = selective_search(img, args.strategy)
        gt_bboxes = parse_annotation(os.path.join(anno_dir, img_id + ".xml"))
        iou_bboxes = []  # proposals with IoU greater than 0.5

        ##################################################
        # TODO: For all the gt_bboxes in each image,     #
        #       please calculate the recall of the       #
        #       gt_bboxes according to the document.     #
        #       Store the bboxes with IoU >= 0.5         #
        #       If there is more than one proposal has   #
        #       IoU >= 0.5 with a same groundtruth bbox, #
        #       store the one with biggest IoU.          #
        ##################################################


        for i in range(len(gt_bboxes)):
            curr = 0 
            for j in proposals:
                iou_val = bb_intersection_over_union(gt_bboxes[i],j) #get iou values

                if iou_val >= 0.5: # check if greater than or equal to 0.5
                    if curr == 0:
                        iou_bboxes.append(j)
                        curr = iou_val
                    elif curr < iou_val: # if there are multiple iou with >0.5, get the best
                        curr = iou_val
                        iou_bboxes.pop()
                        iou_bboxes.append(j)



            recall = len(iou_bboxes)/len(gt_bboxes) # calculate recall
    
        print("Recall for image"+img_name ,recall)





        ##################################################
        # End of TODO                                    #
        ##################################################
        vis_img = img.copy()
        vis_img = visualize(vis_img, gt_bboxes, (255, 0, 0),"gt")
        cv2.imshow("gt", vis_img)
        save = "gt"+str(imgno)+".jpg"
        cv2.imwrite(save, vis_img)
        vis_img = visualize(vis_img, iou_bboxes, (0, 0, 255),"iou")
        cv2.imshow("iou", vis_img)
        save = "iou"+str(imgno)+".jpg"
        cv2.imwrite(save, vis_img)


        proposals_img = img.copy()
        proposals_img = visualize(proposals_img, gt_bboxes, (255, 0, 0),"gt")
        proposals_img = visualize(proposals_img, proposals, (0, 0, 255),"proposal")
        cv2.imshow("proposal", proposals_img)
        save = "proposal"+str(imgno)+".jpg"
        cv2.imwrite(save, proposals_img)
        imgno +=1

        ##################################################
        # TODO: (optional) You may use cv2 to visualize  #
        #       or save the image for report.            #
        ##################################################
        






        ##################################################
        # End of TODO                                    #
        ##################################################
        


if __name__ == "__main__":
    main()




