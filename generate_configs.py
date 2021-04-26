import os
import yaml
import copy

graph_models = ["gat","gcn","monet","multigat"]
image_models = ["cnn","vgg","prevgg"]
datasets = ["mnist","fmnist","cifar10","cifar100"]
datasets_classes = ["mnist_img_slic","mnist_img_slic","cifar_img_slic","cifar_img_slic"]
num_class = [10,10,10,100]
name_to_class = dict(zip(datasets,datasets_classes))
name_to_num_class = dict(zip(datasets, num_class))

graph_img_dir = "configs/custom_trainer/graph_image"
graph_dir = "configs/custom_trainer/graph"
image_dir = "configs/custom_trainer/image"

# Image and Graph

dataset_paths = './configs/templates/dataset/paths'
def generate_dataset_yaml_indiv(typ, model, dataset):
    temp = f"{dataset}.txt"
    if typ=='image':
        transform_path = img_transforms_path
    else:
        transform_path = grp_transforms_path

    transform_path = os.path.join(transform_path, dataset)
    list_models = os.listdir(transform_path)
    list_models = [x.split('.')[0] for x in list_models]
    if model in list_models:
        model_transform = os.path.join(transform_path,f"{model}.txt")
    else:
        model_transform = os.path.join(transform_path,"regular.txt")
    with open(model_transform) as f:
        head = f.read()

    with open(os.path.join(dataset_paths, temp)) as f:
        tail = f.read()

    return head+"\n"+tail
    
def generate_model_yaml_indiv(typ, model, dataset):
    if typ=='image':
        transform_path = img_transforms_path
    else:
        transform_path =  grp_transforms_path

    transform_path = os.path.join(transform_path, model)
    list_datasets = os.listdir(transform_path)
    list_datasets = [x.split('.')[0] for x in list_datasets]
    if dataset in list_datasets:
        model_template = os.path.join(transform_path,f"{dataset}.txt")
    else:
        model_template = os.path.join(transform_path,"regular.txt")
    with open(model_template) as f:
        head = f.read()

    return head

def generate_model_yaml(image_path, graph_path, dataset):
    with open(image_path) as f:
        d1 = yaml.safe_load(f)

    with open(graph_path) as f:
        d2 = yaml.safe_load(f)


    if 'linear_layer_params' in d1:
        d1['linear_layer_params']['intermediate_layer_sizes']=[]
    d1['num_classes']=32

    if 'linear_layer_params' in d2:
        d2['linear_layer_params']['intermediate_layer_sizes']=[]
    d2['num_classes']=32

    d = {'name':'projection'}
    d['cnn_config'] = d1
    d['gnn_config'] = d2
    d['num_classes'] = name_to_num_class[dataset]

    return d

def generate_dataset_yaml(image_path, graph_path, dataset):
    with open(image_path) as f:
        d = yaml.safe_load(f)

    with open(graph_path) as f:
        d2 = yaml.safe_load(f)

    d3 = copy.deepcopy(d)
    d['main']['name'] = name_to_class[dataset]
    d['main']['image_transform_args'] = d3['main']['transform_args']
    del d['main']['transform_args']
    d['main']['graph_transform_args'] = d2['main']['transform_args']

    d['train']['name'] = name_to_class[dataset]
    d['train']['image_transform_args'] = d3['train']['transform_args']
    del d['train']['transform_args']
    d['train']['graph_transform_args'] = d2['train']['transform_args']

    d['val']['name'] = name_to_class[dataset]
    d['val']['image_transform_args'] = d3['val']['transform_args']
    del d['val']['transform_args']
    d['val']['graph_transform_args'] = d2['val']['transform_args']

    d['train_val']['train']['name'] = name_to_class[dataset]
    d['train_val']['train']['image_transform_args'] = d3['train']['transform_args']
    del d['train_val']['train']['transform_args']
    d['train_val']['train']['graph_transform_args'] = d2['train']['transform_args']

    d['train_val']['val']['name'] = name_to_class[dataset]
    d['train_val']['val']['image_transform_args'] = d3['val']['transform_args']
    del d['train_val']['val']['transform_args']
    d['train_val']['val']['graph_transform_args'] = d2['val']['transform_args']

    return d
    
# Image and Graph Dataset
img_transforms_path = './configs/templates/dataset/transforms/image'
grp_transforms_path = './configs/templates/dataset/transforms/graph'

for dataset in datasets:
    for image_model in image_models:
        image_path = os.path.join(image_dir, image_model+'_'+dataset)
        with open(os.path.join(image_path, 'dataset.yaml'),'w') as f:
            f.write(generate_dataset_yaml_indiv('image', image_model, dataset))

    for graph_model in graph_models:
        graph_path = os.path.join(graph_dir, graph_model+'_'+dataset)
        with open(os.path.join(graph_path, 'dataset.yaml'),'w') as f:
            f.write(generate_dataset_yaml_indiv('graph', graph_model, dataset))


# Image and Graph Model
img_transforms_path = './configs/templates/model/image'
grp_transforms_path = './configs/templates/model/graph'

for dataset in datasets:
    for image_model in image_models:
        image_path = os.path.join(image_dir, image_model+'_'+dataset)
        with open(os.path.join(image_path, 'model.yaml'),'w') as f:
            f.write(generate_model_yaml_indiv('image', image_model, dataset))

    for graph_model in graph_models:
        graph_path = os.path.join(graph_dir, graph_model+'_'+dataset)
        with open(os.path.join(graph_path, 'model.yaml'),'w') as f:
            f.write(generate_model_yaml_indiv('graph', graph_model, dataset))

# Image and Graph Train
with open("./configs/templates/train/image.yaml") as f:
    image_train_yaml = f.read()
for root, dirs, fils in os.walk('configs/custom_trainer/image'):
    for fil in fils:
        if 'train.yaml' in fil:
            with open(os.path.join(root, fil),'w') as f:
                f.write(image_train_yaml)

with open("./configs/templates/train/graph.yaml") as f:
    graph_train_yaml = f.read()
for root, dirs, fils in os.walk('configs/custom_trainer/graph'):
    for fil in fils:
        if 'train.yaml' in fil:
            with open(os.path.join(root, fil),'w') as f:
                f.write(graph_train_yaml)


# Graph - Image Entire
with open("./configs/templates/train/graph_image.yaml") as f:
    graph_image_train_yaml = f.read()

for dataset in datasets:
    for image_model in image_models:
        image_path = os.path.join(image_dir, image_model+'_'+dataset)
        for graph_model in graph_models:
            graph_path = os.path.join(graph_dir, graph_model+'_'+dataset)
            graph_image_path = os.path.join(graph_img_dir, image_model+'_'+graph_model+'_'+dataset)
            print(graph_path)
            print(image_path)
            if not os.path.exists(graph_image_path):
                os.makedirs(graph_image_path)
            with open(os.path.join(graph_image_path, 'train.yaml'),'w') as f:
                f.write(graph_image_train_yaml)
            model_yaml = generate_model_yaml(os.path.join(image_path,'model.yaml'),os.path.join(graph_path,'model.yaml'),dataset)
            with open(os.path.join(graph_image_path, 'model.yaml'),'w') as f:
                yaml.dump(model_yaml,f)
            dataset_yaml = generate_dataset_yaml(os.path.join(image_path,'dataset.yaml'),os.path.join(graph_path,'dataset.yaml'),dataset)
            with open(os.path.join(graph_image_path, 'dataset.yaml'),'w') as f:
                yaml.dump(dataset_yaml,f)