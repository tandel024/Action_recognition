import numpy as np
import cv2
import tensorflow as tf
from utils.configurations import KEYPOINT_DICT as kp_names
from utils.predictor import preprocess, get_prediction
from utils.drawing import draw_kps
from utils.math import Distance, ManhattanDistance, XDistance, YSign, YDistance, t4YSign
from argparse import ArgumentParser
from time import time

args = ArgumentParser()
args.add_argument('-v', '--visualize', default=True, action='store_true')
# args.add_argument('-v', '--visualize', action = 'store_true')
args.add_argument('-log', '--log', action='store_true')
args = args.parse_args()


videofile = r"trial4-vet2-Traffic Lights - Fast.mp4"
#videofile= r"vet3-Trial2-RG--Slow.mp4"
adult = str(input('Is the adult video? Enter [y]/n: ')).lower() in 'yes' #"yes"
trial = str(input('Enter the trial no: ')) #"3"
fast = str(input('Is the fast video? Enter [y]/n: ')).lower() in 'yes'
print(fast)

score_trials = {
    '1': {
        'sequence': [
            'pass',
            'pass',
            'pass',
            'pass',
            'pass',
            'pass',
            'pass',
            'pass'
        ],
        'fast_time': [4, 5, 6, 7, 8, 10, 11, 12],
        'slow_time': [6, 8, 10, 11, 13, 15, 17, 19],
        'total_time': 12 if fast else 20
        },
    '2': {
        'sequence': [
            'pass',
            'pass',
            'pass',
            'pass',
            'no-pass',
            'no-pass',
            'pass',
            'pass',
            'no-pass',
            'pass',
            'pass',
            'pass',
            'no-pass',
            'no-pass',
            'pass',
            'no-pass'
        ],
        'fast_time': [5,6,7,8,10,11,12,14,15,16,18,19,20,21,22,23],
        'slow_time': [6,8,10,11,13,15,16,17,19,21,23,24,26,28,30,32,34],
        'total_time': 53 if fast else 37
    },
    '3': {
        'sequence': [
            'pass',
            'pass',
            'up-down',
            'pass',
            'no-pass',
            'up-down',
            'no-pass',
            'up-down',
            'pass',
            'pass',
            'up-down',
            'no-pass',
            'no-pass',
            'up-down',
            'up-down',
            'pass'
        ],
        'fast_time': [4,5,6,7,9,10,11,12,14,15,16,17,18,19,20,22],
        'slow_time': [6,7,9,11,14,15,17,19,20,22,24,26,28,29,31,33],
        'total_time': 23 if fast else 34
    },
    '4': {
        'sequence': [
            'pass',
            'pass',
            'up-down',
            'pass',
            'no-pass',
            'up-down',
            'no-pass',
            'up-down',
            'pass',
            'pass',
            'up-down',
            'no-pass',
            'no-pass'
            'up-down',
            'up-down',
            'pass'
        ],
        'fast_time': [5,6,7,8,10,11,12,14,15,17,18,19,20,22,23,24],
        'slow_time': [5,7,9,12,14,16,19,21,23,25,28,30,32,34,37,39],
        'total_time': 75 if fast else 60
    }
}


seq_array_test = score_trials[trial]['sequence']
fast_time = score_trials[trial]['fast_time']
slow_time = score_trials[trial]['slow_time']

# Initialize the TFLite interpreter
interpreter = None

def pop_max_and_find_difference(lst):
    if not lst:
        return None

    max_value = max(lst)
    lst.remove(max_value)

    if not lst:
        return 0

    fmax = max(lst)
    fmin = min(lst)

    return fmax - fmin

def build_interpreter(path):
    global interpreter
    interpreter = tf.keras.models.load_model(path)
    interpreter = interpreter.signatures["serving_default"]


def create_closer(seq):
    closer_arr = dict()
    for s in seq:
        closer_arr.update({f'left_{s}': 0, f'right_{s}': 0})
    return closer_arr


def scale_up(origin_vec, scale_vec):
    d = scale_vec - origin_vec
    ext = (d + d/2.5) + origin_vec
    return ext



st_time = time()

score_1 = 0
score_2 = 0
score_3 = 0

max_score_1_trial1=8
max_score_2_trial1=16

max_score_1_trial2=10
max_score_2_trial2=20
max_score_3_trial2=12

max_score_1_trial34=12
max_score_2_trial34=24
max_score_3_trial34=8




score_1_seq_array = seq_array_test.copy()
total_time = score_trials[trial]['total_time']
cap = cv2.VideoCapture(videofile)
total_frames = 0
while True:
    ok, _ = cap.read()
    if not ok:
        break
    total_frames += 1

fps = int(round(total_frames/total_time))

build_interpreter(path=r'model')
cap = cv2.VideoCapture(videofile)

threshold = 0
start_frame = fps * 0
frame_count = 0
skip_frames = 5


width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
input_size = (512, 512)

if not cap.isOpened():
    raise IOError("video read fail")

counter = None
detection_directions = list()

hand = None
xthreshold = 130
xthreshold_trial2 = 25
ythreshold_trial3=30

consider_time = np.array(fast_time if fast else slow_time)
# consider_time = np.array(adult_time if adult else kids_time)
time_scores_counted = np.zeros(len(consider_time))
tempXDist = width/2
tempYDist =height/2
tempYDist_d=0
tempYSign = 1
b_flag = False
time_bias = 0
yb_flag=False
no_actioncnt=0
round=0

y1dist=[]
xy1dist=[]

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        break
    if not ((frame_count % skip_frames == 0) and (frame_count/fps > consider_time[0]) and (frame_count > start_frame)):
        continue
    input_tensor, frame, (left, top) = preprocess(frame[...,::-1], input_size=input_size)
    fw, fh, _ = frame.shape
    kpts, scores, classes, boxes, kpts_scores = get_prediction(input_tensor, interpreter, from_class=1)
    kpts = kpts[...,::-1]
    kpts = kpts * np.array([fw, fh])
    preds = np.concatenate([kpts, kpts_scores.reshape(1, 17, 1)], axis=2).squeeze(0)
    left_hand = preds[kp_names['left_wrist']][:2]
    right_hand = preds[kp_names['right_wrist']][:2]
    left_elbow = preds[kp_names['left_elbow']][:2]
    right_elbow = preds[kp_names['right_elbow']][:2]
    left_hand = scale_up(scale_vec=left_hand, origin_vec=left_elbow)
    right_hand = scale_up(scale_vec=right_hand, origin_vec=right_elbow)
    preds[kp_names['left_wrist'], :2] = left_hand
    preds[kp_names['right_wrist'], :2] = right_hand
    distance = Distance(preds[kp_names['right_wrist']][:2], preds[kp_names['left_wrist']][:2])
    xdistance = XDistance(preds[kp_names['right_wrist']], preds[kp_names['left_wrist']])
    ydistance=YDistance(preds[kp_names['right_wrist']], preds[kp_names['left_wrist']])
    ysign = YSign(preds[kp_names['right_wrist']], preds[kp_names['left_wrist']])
    t4ysign=t4YSign(preds[kp_names['right_wrist']], preds[kp_names['left_wrist']])
    
    y1dist.append(ydistance)
    xy1dist.append(distance)
    print("ydistance:",ydistance, "xydistance:",distance, "no_action:", no_actioncnt,"Frame count", frame_count,"FPS:",fps,"time bias",time_bias )
    # print(xdistance,tempXDist,ydistance,tempYDist,tempYDist_d,ysign,t4ysign,b_flag)
    # print(ydistance,tempYDist,tempYDist_d,ysign,yb_flag)
    round+=1

    if args.visualize:

        frame = draw_kps(frame, preds)
        if top > 0:
            frame = frame[top-1:-(top-1)]
        if left > 0:
            frame = frame[:, left-1:-(left-1)]
        if ((ysign*tempYSign<0 or (tempYSign<0 and ysign<0)) and b_flag==False and trial!="1" and trial!="4"):
            if fast:
                if round > 1:
                    tempXDist = width/2
                    tempYDist= height/2
                    tempYSign = ysign
                    b_flag = True
            else:
                if round>4:
                    tempXDist = width / 2
                    tempYDist = height / 2
                    tempYSign = ysign
                    b_flag = True

        if ((ysign*tempYSign<0 or (tempYSign<0 and ysign<0)) and round>3 and b_flag==False and trial=="4"):
            tempXDist = width/2
            tempYDist= height/2
            tempYDist_d = 0
            tempYSign = ysign
            b_flag = True

        if round>5 and b_flag==False:
            tempXDist = width/2
            tempYDist= height/2
            tempYSign = ysign
            b_flag = True

        if (ysign*tempYSign<0 and b_flag==False and trial=="1"):
            tempXDist = width/2
            tempYDist= height/2
            tempYSign = ysign
            b_flag = True

        if (tempXDist<xdistance and b_flag and xdistance<xthreshold and score_1 <len(consider_time) and trial=="1"):
            score_1 += 1
            score_2+=2
            cv2.putText(frame, "pass" + str(score_1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "Score_1:"+str(score_1), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score_2:"+str(score_2), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

            b_flag = False
            # tempXDist = width/2

            if score_1 == 1:
                time_bias = frame_count/fps - consider_time[0]
            print("pass ball","score_1:",score_1,"score_2",score_2,"score_3:",score_3)
            if (frame_count/fps > consider_time[score_1-1]+time_bias and frame_count/fps < consider_time[score_1-1]+0.5+time_bias):
                score_2 += 2
            # print(score_1, xdistance)


        if (tempYDist<ydistance and b_flag and tempXDist<xthreshold_trial2 and xdistance<xthreshold_trial2+16 and score_1 <max_score_1_trial2 and trial=="2"):
            score_1 += 1
            score_2+=2
            cv2.putText(frame, "pass" + str(score_1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "Score_1:"+str(score_1), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score_2:"+str(score_2), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            b_flag = False
            # tempXDist = width/2

            if score_1 == 1:
                time_bias = frame_count/fps - consider_time[0]
            # print(score_1, score_2, score_3)
            if (frame_count/fps > consider_time[score_1-1]+time_bias and frame_count/fps < consider_time[score_1-1]+0.5+time_bias):
                score_2 += 2

            print("pass ball","score_1:",score_1,"score_2",score_2,"score_3:",score_3)
            print ("ydl",y1dist)
            round=0
            no_actioncnt=0
            y1dist.clear()
            xy1dist.clear()



        if (tempXDist<xdistance and tempYDist<ydistance and b_flag and tempXDist<xthreshold_trial2 and score_1 <max_score_1_trial34 and trial=="3" and score_2<max_score_2_trial34):
            score_1 += 1
            score_2+=2
            cv2.putText(frame, "pass" + str(score_1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "Score_1:"+str(score_1), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score_2:"+str(score_2), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            b_flag = False
            # tempXDist = width/2
            if score_1 == 1:
                time_bias = frame_count/fps - consider_time[0]
            # print(score_1, score_2,score_3)
            
            print("pass ball","score_1:",score_1,"score_2",score_2,"score_3:",score_3)
            print ("ydl",y1dist)
            round=0
            no_actioncnt=0
            y1dist.clear()
            xy1dist.clear()


        if (5<tempYDist_d and yb_flag==False and ((tempXDist>xdistance and tempYDist<ydistance and ydistance>10)
             or(tempXDist<xdistance and tempYDist<ydistance)) and b_flag and ydistance<40 and xdistance<xthreshold_trial2+21 and score_1 <max_score_1_trial34 and trial=="4"):
            score_1 += 1
            score_2+=2
            cv2.putText(frame, "pass" + str(score_1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "Score_1:"+str(score_1), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score_2:"+str(score_2), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
            b_flag = False
            yb_flag=False
            tempYDist_d = 0
            # tempXDist = width/2
            if score_1 == 1:
                time_bias = frame_count/fps - consider_time[0]

            if (frame_count/fps > consider_time[score_1-1]+time_bias and frame_count/fps < consider_time[score_1-1]+0.5+time_bias):
                score_2 += 2
            print("pass ball","score_1:",score_1,"score_2",score_2,"score_3:",score_3)
            # round=0
            no_actioncnt=0



        if (tempYDist_d > ythreshold_trial3 and yb_flag and ydistance < ythreshold_trial3 and score_1 < max_score_1_trial34 and score_2<max_score_2_trial34 and trial == "3"):
            score_1 += 1
            score_2 += 2
            if(score_3<max_score_3_trial34):
                score_3 += 1
            cv2.putText(frame, "up_down" + str(score_1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "Score_1:"+str(score_1), (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score_3:"+str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score_2:" + str(score_2), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            # tempXDist = width/2
            yb_flag = False
            no_actioncnt = 0
            tempYDist_d = 0

            print("up_down","score_1:",score_1,"score_2",score_2,"score_3:",score_3)
            y1dist.clear()
            xy1dist.clear()

        if (tempYDist_d > ythreshold_trial3 and b_flag and yb_flag and ydistance < ythreshold_trial3+10 and score_1 < max_score_1_trial34 and trial == "4"):
            
            score_1+=1
            score_2 += 2
        
            if(score_3<max_score_3_trial34):
                score_3 += 1
            cv2.putText(frame, "up_down" + str(score_1), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(frame, "Score_1:"+str(score_1), (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score_2:" + str(score_2), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,cv2.LINE_AA)
            # tempXDist = width/2
            yb_flag = False
            b_flag=False
            # b_flag=False
            tempYDist_d = 0
            round=0
            no_actioncnt=0
            print("up_down","score_1:",score_1,"score_2",score_2,"score_3:",score_3)

        #Trial 2 score 3 calculation
        if score_1>0 and b_flag and score_3<max_score_3_trial2 and trial=="2":
    
            if fast==False :
                if no_actioncnt > 10:
                    
                        if pop_max_and_find_difference(y1dist) > 11.0 and pop_max_and_find_difference(xy1dist) > 11.0:
                         score_3 +=1
                         cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                     cv2.LINE_AA)
                         no_actioncnt=0

                         print ('HAND MOVED MISTAKE, score_3 : ', score_3 )
                         y1dist.clear()

                        else:
                             score_3+=2
                             cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                     cv2.LINE_AA)
                             no_actioncnt=0
                             print("no move","score_3", score_3)
                             print ("ydl",y1dist)
                             y1dist.clear()
            
            if fast==True:
                
                if no_actioncnt > 6:
                    if pop_max_and_find_difference(y1dist) > 11.0 and pop_max_and_find_difference(xy1dist) > 11.0:
                         score_3 +=1
                         cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                     cv2.LINE_AA)
                         no_actioncnt=0
                         print ('HAND MOVED MISTAKE, score_3 : ', score_3 )
                         y1dist.clear()

                    else:
                             score_3+=2
                             cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                     cv2.LINE_AA)
                             no_actioncnt=0
                             print("no move","score_3", score_3)
                             print ("ydl",y1dist)
                             y1dist.clear()
        


#Trial 3 score_3 calculation
        if ydistance<11 and score_1>0 and b_flag and score_3<max_score_3_trial34 and (trial=="3"):
            if fast==True:
                if no_actioncnt >11:
                    if no_actioncnt>13:
                        score_3+=2
                    if pop_max_and_find_difference(y1dist) > 50.0 and pop_max_and_find_difference(xy1dist) > 12.0:
                         score_3 +=1
                         cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                     cv2.LINE_AA)
                         no_actioncnt=0

                         print ('HAND MOVED MISTAKE, score_3 : ', score_3 )
                         y1dist.clear()

                    else:
                        score_3+=2
                        cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,cv2.LINE_AA)
                        no_actioncnt=0
                        print("no move","score_3", score_3)
                        print ("ydl",y1dist)
                        y1dist.clear()
            elif fast==False:
                if no_actioncnt > 13:
                    if no_actioncnt > 16:
                        score_3+=2
                    if pop_max_and_find_difference(y1dist) > 11.0 and pop_max_and_find_difference(xy1dist) > 9.0:
                         score_3 +=1
                         cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                     cv2.LINE_AA)
                         no_actioncnt=0
                         print ('HAND MOVED MISTAKE, score_3 : ', score_3 )
                         y1dist.clear()

                    else:
                             score_3+=2
                             cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                     cv2.LINE_AA)
                             no_actioncnt=0
                             print("no move","score_3", score_3)
                             print ("ydl",y1dist)
                             y1dist.clear()
#trial 4 score 3 calculation
        if ydistance<10 and score_1>0 and b_flag and score_3<max_score_3_trial34-2 and  trial=="4":
            if fast==True:
                if no_actioncnt >9:
                    score_3+=2
                    cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                cv2.LINE_AA)
                    no_actioncnt=0
                    print("no move","score_3", score_3)
            elif fast==False:
                if no_actioncnt > 15:
                    score_3+=2
                    cv2.putText(frame, "Score_3:" + str(score_3), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1,
                                cv2.LINE_AA)
                    no_actioncnt=0
                    print("no move","score_3", score_3)
        if (b_flag==True and yb_flag==False) or (round>3 and trial=="2") :
            no_actioncnt += 1


        if(tempYDist_d>40):
            yb_flag=True

        elif trial=="3" and (tempYDist_d>30):
            yb_flag = True

        if(tempYDist_d < ydistance):
            tempYDist_d = ydistance

        if (tempXDist>xdistance):
            tempXDist = xdistance

        if (tempYDist > ydistance):
            tempYDist=ydistance
            # b_flag = True
        cv2.imshow('Frame', frame[...,::-1])
        cv2.waitKey(1)

    preds = dict(zip(kp_names, preds))

    


cap.release()
if args.visualize:
    cv2.destroyAllWindows()

print(f'''
Final Calculated Scores are:
    SCORE 1: {score_1}
    SCORE 2: {score_2}
    SCORE 3: {score_3}
    Total:   {score_1 + score_2 + score_3}
''')

print("Total time taken:", int((time() - st_time)/60), 'mins')





