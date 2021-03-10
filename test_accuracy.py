# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:37:29 2020

@author: Maxime Gardoni, Hila Vardi, Niccolò Stefanini
"""

import modules as M

import dlc_practical_prologue as prologue


mini_batch_size = 100
N = 1000    #size of dataset
lr= 1e-3    #learning rate

'''----------------------------  Get dataset  ------------------------------'''
#get dataset
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
#double training set
train_input, train_classes, train_target = M.doubleTrainSet(train_input, train_classes, train_target)
#normalization
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
test_input.sub_(mu).div_(std)

#%%
'''-------------------  Transfer Learning - calling  -----------------------'''
print("training: Transfer Learning")
model = M.transferLearnNet(batch_normalization = False)
M.train_model_transferL(model, train_input[:, 0:1, :, :], train_classes[:, 0], 
                      train_target, mini_batch_size, lr, "adam")
test_acc = 1-M.compute_err_perc_transferL(model, test_input, test_target, 
                                        mini_batch_size, "compar")

print("Transfer Learning : test accuracy = {:.04f}".format(test_acc))

#%%
'''----------------  Siamese Network no WS - calling  ----------------------'''
print("training: Siamse Network no WS")
model = M.SiamNetDigitNoWeightShare( False, False, False )
M.train_siam_model(model, train_input, train_classes, train_target, 
                 mini_batch_size, lr, "adam", False)
test_acc = 1-M.compute_err_perc_siam(model, test_input, test_target, 
                                   mini_batch_size, outputIDToCheck="compar")
        
print("Siamese Network no WS : test accuracy = {:.04f}".format(test_acc))

#%%
'''------------------  Siamese Network WS - calling  -----------------------'''
print("training: Siamse Network WS")
model = M.SiamNetDigit( False, False, False )
M.train_siam_model(model, train_input, train_classes, train_target, 
                 mini_batch_size, lr, "adam", False)
test_acc = 1-M.compute_err_perc_siam(model, test_input, test_target, 
                                   mini_batch_size, outputIDToCheck="compar")
        
print("Siamese Network WS : test accuracy = {:.04f}".format(test_acc))

#%%
'''----------------  Siamese Network WS,AUX - calling  ---------------------'''
print("training: Siamse Network WS,AUX")
model = M.SiamNetDigit( True, False, False )
M.train_siam_model(model, train_input, train_classes, train_target, 
                 mini_batch_size, lr, "adam", False)
test_acc = 1-M.compute_err_perc_siam(model, test_input, test_target, 
                                   mini_batch_size, outputIDToCheck="compar")
        
print("Siamese Network WS,AUX : test accuracy = {:.04f}".format(test_acc))

#%%
'''--------------  Siamese Network WS,AUX,BN - calling  --------------------'''
print("training: Siamse Network WS,AUX,BN")
model = M.SiamNetDigit( True, True, False )
M.train_siam_model(model, train_input, train_classes, train_target, 
                 mini_batch_size, lr, "adam", False)
test_acc = 1-M.compute_err_perc_siam(model, test_input, test_target, 
                                   mini_batch_size, outputIDToCheck="compar")
        
print("Siamese Network WS,AUX,BN : test accuracy = {:.04f}".format(test_acc))

#%%
'''-----------  Siamese Network WS,AUX,Dropout - calling  ---------------'''
print("training: Siamse Network WS,AUX,Dropout")
model = M.SiamNetDigit( True, False, True )
M.train_siam_model(model, train_input, train_classes, train_target, 
                 mini_batch_size, lr, "adam", False)
test_acc = 1-M.compute_err_perc_siam(model, test_input, test_target, 
                                   mini_batch_size, outputIDToCheck="compar")
        
print("Siamese Network WS,AUX,Dropout : test accuracy = {:.04f}".format(test_acc))







