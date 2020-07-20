import wget
import socket
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format, json_format
from tensorflow import keras
import pandas as pd

import os

logdir = '/home/ubuntu/diploma/logs'
if not os.path.exists(logdir):
    os.makedirs(logdir)

def run_tensorboard():
    from tensorboard import program
    from tensorboard import default
    from tensorboard.compat import tf
    from tensorboard.plugins import base_plugin
    from tensorboard.uploader import uploader_subcommand
    
    tb = program.TensorBoard(default.get_plugins() + default.get_dynamic_plugins(),
#         program.get_default_assets_zip_provider(),
        subcommands=[uploader_subcommand.UploaderSubcommand()])
    
    tb.configure(argv=[None, '--logdir', logdir, '--port', '6006'])
    url = tb.launch()
    print(url)

run_tensorboard()
    
def _from_url(url, keep = False):
    out = './mygraph.pbtxt'
    os.system('rm -f {}'.format(out))
    wget.download(url,out)

    gdef = graph_pb2.GraphDef()

    with open(out, 'r') as f:
        graph_str = f.read()

    mygraph_msg = text_format.Parse(graph_str, gdef)

    graph_dict = json_format.MessageToDict(mygraph_msg)
    
    if keep == False:
        os.system('rm {}'.format(out))
        
    return graph_dict


def tf_graph_dict(net, keep = False):
    
    os.system('rm -rf {}'.format(logdir))
        
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

    tensorboard_callback.set_model(net)
        
    return _from_url('http://localhost:6006/data/plugin/graphs/graph?run=train', keep = keep)

def torch_graph_dict(model, train_loader, keep = False):
    from torch.utils.tensorboard import SummaryWriter

    out = '{}/train/'.format(logdir)
    os.system('rm -rf {}').format(logdir)

    writer = SummaryWriter(out)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    writer.add_graph(model, images)
    writer.close()
    
    return _from_url('http://localhost:6006/data/plugin/graphs/graph?run=train', keep = keep)

pred_epochs = 1
pred_start = 10
pred_stop = 100

def just_tf(net, x_train, y_train, BATCH):
    
    net.fit(x_train, y_train, batch_size = BATCH,
                epochs=pred_epochs, steps_per_epoch = pred_stop)
    
def tf(net, x_train, y_train, BATCH):
    
    host = socket.gethostname()
    
    os.system('rm -rf {}'.format(logdir))
    
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, profile_batch = (pred_start, pred_stop) )

    net.fit(x_train, y_train, batch_size = BATCH,
                epochs=pred_epochs, steps_per_epoch = pred_stop,
                callbacks=[tensorboard_callback])

    dire = '{}/train/plugins/profile'.format(logdir)
    [entry] = os.listdir(dire)

    os.rename(os.path.join(dire,entry),(os.path.join(dire, 'my')))
    url = 'http://localhost:6006/data/plugin/profile/data?run=train/my&tag=tensorflow_stats&host={}&tqx=out:csv;'.format(host)

    out = './prof.csv'
    if os.path.exists(out):
        os.remove(out)
    wget.download(url,out)

    df = pd.read_csv(out, index_col=0)
    
    os.remove(out)

    df = df[['Type', 'Operation', '#Occurrences', 'Avg. self-time (us)']]
#     df = df[df['#Occurrences'] > 1]
    df = df.sort_values(by = ['Operation', 'Type']).reset_index(drop=True)
    
    return df
