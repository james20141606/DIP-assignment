#!/usr/bin/env python

import urllib.request
import urllib.error
import time
import json
import numpy as np

http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
key = "1D6VX9riO_EZNgZIM3komrFM4y8No6lu"
secret = "gTEbchrwsadPg2PKvN3OMTsuqpkFjz3G"

def get_key_points(filepath):
#filepath = 'source1.png'

    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filepath, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('1')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(
        "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')

    http_body = b'\r\n'.join(data)

    # build http request
    req = urllib.request.Request(url=http_url, data=http_body)

    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        qrcont = resp.read()
        # if you want to load as json, you should decode first,
        # for example: json.loads(qrount.decode('utf-8'))
        #print(qrcont.decode('utf-8'))
        return np.array([[value['y'],value['x']] for value in json.loads(qrcont)['faces'][0]['landmark'].values()])
        #len(json.loads(qrcont)['faces'][0]['landmark'].keys()) 83
        #detected_points = np.array([[value['y'],value['x']] for value in json.loads(qrcont)['faces'][0]['landmark'].values()])
    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))
    

def get_frame_key_points(image,points):
    width=image.shape[0]
    height=image.shape[1]
    add_points = np.array([0,0,width,0,0,height,width/2,0,width/2,height,
          0,height/2,width,height/2,width,height]).astype('int').reshape(-1,2)
    return np.concatenate((points,add_points))

if __name__=='__main__':
    detected_points_source1 = get_key_points('source1.png')
    detected_points_source2 = get_key_points('source2.png')
    detected_points_target1 = get_key_points('target1.png')

    detected_points_source1 = get_frame_key_points(source1,detected_points_source1)
    detected_points_source2 = get_frame_key_points(source2,detected_points_source2)
    detected_points_target1 = get_frame_key_points(target1,detected_points_target1)
    #detected_points_target2 = get_frame_key_points(target2,detected_points_target2)
    np.savetxt('detected_points_source1.txt',detected_points_source1)
    np.savetxt('detected_points_source2.txt',detected_points_source2)
    np.savetxt('detected_points_target1.txt',detected_points_target1)
    #np.savetxt('detected_points_target2.txt',detected_points_target2)