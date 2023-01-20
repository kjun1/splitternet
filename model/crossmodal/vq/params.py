import math

params = dict()

# UttrEnc setting
params['UTTR_ENC_NUM_LAYERS'] = 4

params['UTTR_ENC_CONV1_CHANNELS'] = 1
params['UTTR_ENC_CONV2_CHANNELS'] = 16
params['UTTR_ENC_CONV3_CHANNELS']= 32
params['UTTR_ENC_CONV4_CHANNELS'] = 32
params['UTTR_ENC_CONV5_CHANNELS'] = 16

params['UTTR_ENC_CONV1_KERNEL'] = (3, 9)
params['UTTR_ENC_CONV2_KERNEL'] = (4, 8)
params['UTTR_ENC_CONV3_KERNEL'] = (4, 8)
params['UTTR_ENC_CONV4_KERNEL'] = (9, 5)

params['UTTR_ENC_CONV1_STRIDE'] = (1, 1)
params['UTTR_ENC_CONV2_STRIDE'] = (2, 2)
params['UTTR_ENC_CONV3_STRIDE'] = (2, 2)
params['UTTR_ENC_CONV4_STRIDE'] = (9, 1)

for i in range(1, params['UTTR_ENC_NUM_LAYERS']+1):
    params[f'UTTR_ENC_CONV{i}_PADDING'] = tuple([math.floor((params[f'UTTR_ENC_CONV{i}_KERNEL'][j]-params[f'UTTR_ENC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    

# UttrDec setting
params['UTTR_DEC_NUM_LAYERS'] = 4

params['UTTR_DEC_CONV1_CHANNELS'] = 8
params['UTTR_DEC_CONV2_CHANNELS'] = 16
params['UTTR_DEC_CONV3_CHANNELS']= 16
params['UTTR_DEC_CONV4_CHANNELS'] = 8
params['UTTR_DEC_CONV5_CHANNELS'] = 2

params['UTTR_DEC_CONV1_KERNEL'] = (9, 5)
params['UTTR_DEC_CONV2_KERNEL'] = (4, 8)
params['UTTR_DEC_CONV3_KERNEL'] = (4, 8)
params['UTTR_DEC_CONV4_KERNEL'] = (3, 9)

params['UTTR_DEC_CONV1_STRIDE'] = (9, 1)
params['UTTR_DEC_CONV2_STRIDE'] = (2, 2)
params['UTTR_DEC_CONV3_STRIDE'] = (2, 2)
params['UTTR_DEC_CONV4_STRIDE'] = (1, 1)

for i in range(1, params['UTTR_DEC_NUM_LAYERS']+1):
    params[f'UTTR_DEC_CONV{i}_PADDING'] = tuple([math.ceil((params[f'UTTR_DEC_CONV{i}_KERNEL'][j]-params[f'UTTR_DEC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    params[f'UTTR_DEC_CONV{i}_OUT_PADDING'] = tuple([(params[f'UTTR_DEC_CONV{i}_KERNEL'][j]-params[f'UTTR_DEC_CONV{i}_STRIDE'][j])%2 for j in range(2)])
    
    
# FaceEnc setting
params['FACE_ENC_CONV_LAYERS'] = 5
params['FACE_ENC_LINEAR_LAYERS'] = 2

params['FACE_ENC_CONV1_CHANNELS'] = 3
params['FACE_ENC_CONV2_CHANNELS'] = 32
params['FACE_ENC_CONV3_CHANNELS']= 64
params['FACE_ENC_CONV4_CHANNELS'] = 128
params['FACE_ENC_CONV5_CHANNELS'] = 128
params['FACE_ENC_CONV6_CHANNELS'] = 256

params['FACE_ENC_LINEAR1_CHANNELS'] = 256
params['FACE_ENC_LINEAR2_CHANNELS'] = 16
params['FACE_ENC_LINEAR3_CHANNELS'] = 8

params['FACE_ENC_CONV1_KERNEL'] = (6, 6)
params['FACE_ENC_CONV2_KERNEL'] = (6, 6)
params['FACE_ENC_CONV3_KERNEL'] = (4, 4)
params['FACE_ENC_CONV4_KERNEL'] = (4, 4)
params['FACE_ENC_CONV5_KERNEL'] = (2, 2)

params['FACE_ENC_CONV1_STRIDE'] = (2, 2)
params['FACE_ENC_CONV2_STRIDE'] = (2, 2)
params['FACE_ENC_CONV3_STRIDE'] = (2, 2)
params['FACE_ENC_CONV4_STRIDE'] = (2, 2)
params['FACE_ENC_CONV5_STRIDE'] = (2, 2)



for i in range(1, params['FACE_ENC_CONV_LAYERS']+1):
    params[f'FACE_ENC_CONV{i}_PADDING'] = tuple([math.floor((params[f'FACE_ENC_CONV{i}_KERNEL'][j]-params[f'FACE_ENC_CONV{i}_STRIDE'][j])/2) for j in range(2)])

    
# FaceDec setting
params['FACE_DEC_LINEAR_LAYERS'] = 2
params['FACE_DEC_CONV_LAYERS'] = 5

params['FACE_DEC_LINEAR1_CHANNELS'] = 8
params['FACE_DEC_LINEAR2_CHANNELS'] = 128
params['FACE_DEC_LINEAR3_CHANNELS'] = 2048

params['FACE_DEC_CONV1_CHANNELS']= 128
params['FACE_DEC_CONV2_CHANNELS'] = 128
params['FACE_DEC_CONV3_CHANNELS'] = 64
params['FACE_DEC_CONV4_CHANNELS'] = 32
params['FACE_DEC_CONV5_CHANNELS'] = 6
params['FACE_DEC_CONV6_CHANNELS'] = 6

params['FACE_DEC_CONV1_KERNEL'] = (3, 3)
params['FACE_DEC_CONV2_KERNEL'] = (6, 6)
params['FACE_DEC_CONV3_KERNEL'] = (6, 6)
params['FACE_DEC_CONV4_KERNEL'] = (6, 6)
params['FACE_DEC_CONV5_KERNEL'] = (5, 5)

params['FACE_DEC_CONV1_STRIDE'] = (2, 2)
params['FACE_DEC_CONV2_STRIDE'] = (2, 2)
params['FACE_DEC_CONV3_STRIDE'] = (2, 2)
params['FACE_DEC_CONV4_STRIDE'] = (2, 2)
params['FACE_DEC_CONV5_STRIDE'] = (2, 2) # 元論文の実装と違う



for i in range(1, params['FACE_DEC_CONV_LAYERS']+1):
    params[f'FACE_DEC_CONV{i}_PADDING'] = tuple([math.ceil((params[f'FACE_DEC_CONV{i}_KERNEL'][j]-params[f'FACE_DEC_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    params[f'FACE_DEC_CONV{i}_OUT_PADDING'] = tuple([(params[f'FACE_DEC_CONV{i}_KERNEL'][j]-params[f'FACE_DEC_CONV{i}_STRIDE'][j])%2 for j in range(2)])
    
    
# VoiceDis setting
params['VOICE_DIS_CONV_LAYERS'] = 7
params['VOICE_DIS_LINEAR_LAYERS'] = 2

params['VOICE_DIS_CONV1_CHANNELS'] = 1
params['VOICE_DIS_CONV2_CHANNELS'] = 32
params['VOICE_DIS_CONV3_CHANNELS']= 64
params['VOICE_DIS_CONV4_CHANNELS'] = 128
params['VOICE_DIS_CONV5_CHANNELS'] = 128
params['VOICE_DIS_CONV6_CHANNELS'] = 128
params['VOICE_DIS_CONV7_CHANNELS'] = 64
params['VOICE_DIS_CONV8_CHANNELS'] = 16

params['VOICE_DIS_LINEAR1_CHANNELS'] = 80
params['VOICE_DIS_LINEAR2_CHANNELS'] = 64
params['VOICE_DIS_LINEAR3_CHANNELS'] = 16

params['VOICE_DIS_CONV1_KERNEL'] = (3, 9)
params['VOICE_DIS_CONV2_KERNEL'] = (4, 8)
params['VOICE_DIS_CONV3_KERNEL'] = (4, 8)
params['VOICE_DIS_CONV4_KERNEL'] = (4, 8)
params['VOICE_DIS_CONV5_KERNEL'] = (4, 5)
params['VOICE_DIS_CONV6_KERNEL'] = (1, 5)
params['VOICE_DIS_CONV7_KERNEL'] = (1, 5)

params['VOICE_DIS_CONV1_STRIDE'] = (1, 1)
params['VOICE_DIS_CONV2_STRIDE'] = (2, 2)
params['VOICE_DIS_CONV3_STRIDE'] = (2, 2)
params['VOICE_DIS_CONV4_STRIDE'] = (2, 2)
params['VOICE_DIS_CONV5_STRIDE'] = (4, 1)
params['VOICE_DIS_CONV6_STRIDE'] = (1, 1)
params['VOICE_DIS_CONV7_STRIDE'] = (1, 1)

for i in range(1, params['VOICE_DIS_CONV_LAYERS']+1):
    params[f'VOICE_DIS_CONV{i}_PADDING'] = tuple([math.floor((params[f'VOICE_DIS_CONV{i}_KERNEL'][j]-params[f'VOICE_DIS_CONV{i}_STRIDE'][j])/2) for j in range(2)])
    

#vq params

params['NUM_EMBEDDINGS'] = 16
params['EMBEDDINGS_DIM'] = 8
params['BETA'] = 0.4
    
    
# opt params
params["LR"] = 1e-3
    
params["LAMBDA1"] = 0.01
params["LAMBDA2"] = 0.1
params["LAMBDA3"] = 0.001
params["LAMBDA4"] = 0.01
params["LAMBDA5"] = 1



