import tensorflow as tf
import json

suffix_name  = ""


f = open ('./results/test/predIdData/model_details.json', "r")
data = json.loads(f.read())
for i in data:
    accuracy = i["model_acc"]
    suffix_name = " , ".join([item.split(" : ")[1] for item in i["model_detail"].split(" , ")])
    writer = tf.summary.FileWriter(logdir='./graphs',filename_suffix = suffix_name)
    acc_var = tf.Variable(0, dtype=tf.float32) # variable that holds accuracy
    acc_summ = tf.summary.scalar('Accuracy', acc_var) # summary to write to TensorBoard
    sess = tf.Session()
    for epoch, acc in enumerate(accuracy):
        sess.run(acc_var.assign(acc)) # update accuracy variable
        writer.add_summary(sess.run(acc_summ), epoch+1) # add summary

    writer.flush() # make sure everything is written to disk
    writer.close() # not really needed, but good habit
f.close()