# run_graph.sh before using this script
import wget
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format, json_format
import tensorflow as tf
import os

def run_tensorboard():
    from tensorboard import program
    from tensorboard import default
    from tensorboard.compat import tf
    from tensorboard.plugins import base_plugin
    from tensorboard.uploader import uploader_subcommand
    
    tb = program.TensorBoard(default.get_plugins(),
        subcommands=[uploader_subcommand.UploaderSubcommand()])
    
    tb.configure(argv=[None, '--logdir', './logs', '--port', '6008'])
    url = tb.launch()
    print(url)

def _from_url(url):
    out = './mygraph.pbtxt'
    os.system('rm -f {}'.format(out))
    wget.download(url,out)

    gdef = graph_pb2.GraphDef()

    with open(out, 'r') as f:
        graph_str = f.read()

    mygraph_msg = text_format.Parse(graph_str, gdef)

    graph_dict = json_format.MessageToDict(mygraph_msg)
    
    os.system('rm {}'.format(out))
        
    return graph_dict

def tf_graph_dict(Model):
    logdir = './logs'
    os.system('rm -rf {}'.format(logdir))

    model = Model.model
    data = Model.tf_data.batch(32)
    
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(data, steps_per_epoch = 3, epochs = 1, callbacks=[tb])
        
    return _from_url('http://localhost:6006/data/plugin/graphs/graph?run=train')

def torch_graph_dict(Model):
    out = './logs'
    os.system('rm -rf {}'.format(out))
    
    from torch.utils.tensorboard import SummaryWriter
    
    model, train_loader = Model.model, Model.train_loader

    writer = SummaryWriter(out)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    writer.add_graph(model, images)
    writer.close()
    
    return _from_url('http://localhost:6006/data/plugin/graphs/graph?run=.')
