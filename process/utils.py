# description of the file : 
"""
ABI
helper functions 
like SMOTE (upsample)  
"""
def get_newlabel(row):
    survey = row["survey"]
    if survey == 2:
        return 1
    elif survey == 3:
        return 2
    elif survey == 4 or survey == 5:
        return 3
    return survey


def split_source(y_true,y_pred,test_source):
    mongodb_pred = []
    mongodb_target = []

    react_pred = []
    react_target = []

    socketio_pred = []
    socketio_target = []
    
    for p,t,s in zip(y_pred,y_true,test_source):
        if s[0] == 0:
            mongodb_pred.append(p)
            mongodb_target.append(t)
        elif s[0] == 1:
            react_pred.append(p)
            react_target.append(t)
        else:
            socketio_pred.append(p)
            socketio_target.append(t)
    mongodb = [mongodb_target, mongodb_pred]
    react = [react_target, react_pred]
    socketio = [socketio_target, socketio_pred]
    return {'mongodb':mongodb,
            'react':react,
            'socketio':socketio}