import os, torch
from torchvision.utils import save_image
from torchvision import datasets
try:
    import output as out
    import inverted_obj as io
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')


flower_classes = ['pink primrose', 
'hard-leaved pocket orchid', 
'canterbury bells', 
'sweet pea', 
'english marigold', 
'tiger lily', 
'moon orchid', 
'bird of paradise', 
'monkshood', 
'globe thistle', 
'snapdragon', 
'colt\'s foot', 
'king protea', 
'spear thistle', 
'yellow iris', 
'globe-flower', 
'purple coneflower', 
'peruvian lily', 
'balloon flower', 
'giant white arum lily', 
'fire lily', 
'pincushion flower', 
'fritillary', 
'red ginger', 
'grape hyacinth', 
'corn poppy', 
'prince of wales feathers', 
'stemless gentian', 
'artichoke', 
'sweet william', 
'carnation', 
'garden phlox', 
'love in the mist', 
'mexican aster', 
'alpine sea holly', 
'ruby-lipped cattleya', 
'cape flower', 
'great masterwort', 
'siam tulip', 
'lenten rose', 
'barbeton daisy', 
'daffodil', 
'sword lily', 
'poinsettia', 
'bolero deep blue', 
'wallflower', 
'marigold', 
'buttercup', 
'oxeye daisy', 
'common dandelion', 
'petunia', 
'wild pansy', 
'primula', 
'sunflower', 
'pelargonium', 
'bishop of llandaff', 
'gaura', 
'geranium', 
'orange dahlia', 
'pink-yellow dahlia?', 
'cautleya spicata', 
'japanese anemone', 
'black-eyed susan', 
'silverbush', 
'californian poppy', 
'osteospermum', 
'spring crocus', 
'bearded iris', 
'windflower', 
'tree poppy', 
'gazania', 
'azalea', 
'water lily', 
'rose', 
'thorn apple', 
'morning glory', 
'passion flower', 
'lotus', 
'toad lily', 
'anthurium', 
'frangipani', 
'clematis', 
'hibiscus', 
'columbine', 
'desert-rose', 
'tree mallow', 
'magnolia', 
'cyclamen', 
'watercress', 
'canna lily', 
'hippeastrum', 
'bee balm', 
'ball moss', 
'foxglove', 
'bougainvillea', 
'camellia', 
'mallow', 
'mexican petunia', 
'bromelia', 
'blanket flower', 
'trumpet creeper', 
'blackberry lily']


def get_classes_names(dataset, data_path):
    if dataset == 'imagenet':
        dset = datasets.ImageNet(data_path, split='val')
        ordered_class_labels = []
        for idx, x in enumerate(dset.classes):
            all_names = []
            for i in range(len(x)):
                all_names.append('_'.join(x[i].split()))
            if idx == 134:
                all_names.append('bird')
            elif idx == 517:
                all_names.append('machine')
            ordered_class_labels.append('-'.join(all_names))
        return ordered_class_labels
    elif dataset == 'cifar10':
        dset = datasets.CIFAR10(data_path, train=False)
        return dset.classes
    elif dataset == 'cifar100':
        dset = datasets.CIFAR100(data_path, train=False)
        return dset.classes
    elif dataset == 'oxford-iiit-pets':
        dset = datasets.OxfordIIITPet(root=data_path, split='test')
        return dset.classes
    elif dataset == 'flowers':
        return flower_classes
    else:
        return ['none']

def save_batched_images(full_path, tensors, indices, labels, classes_name, append):
    full_path = os.path.abspath(full_path)
    for _d in out.recursive_create_dir(full_path):
        out.create_dir(_d)
    out.create_dir(full_path)
    for idx, img, label in zip(indices, tensors, labels):
        img_path = f'{full_path}/{int(idx)}_{classes_name[int(label)]}' + append + '.png'
        save_image(img, img_path)

def save_tensor_images(path, img_indices, seed_name, results, seed, targets, labels, classes_name):
    path = os.path.abspath(path)
    for _d in out.recursive_create_dir(path):
        out.create_dir(_d)
    path_target = os.path.join(path, 'target')
    out.create_dir(path_target)
    path_result = os.path.join(path, 'result')
    out.create_dir(path_result)

    if seed is not None:
        save_image(seed, os.path.join(path, f'{seed_name}.png'))
    # io.save_object(0, seed, 0, os.path.join(path, f'{seed_name}.pkl'))
    save_batched_images(path_target, targets, img_indices, labels, classes_name, f'_seed_{seed_name}')
    save_batched_images(path_result, results, img_indices, labels, classes_name, f'_seed_{seed_name}')

    print(f'=> Saved images in {path}')


def save_tensor_reps(path, model2, seed_name, results, targets, rep_type):
    path = os.path.abspath(path)
    for _d in out.recursive_create_dir(path):
        out.create_dir(_d)
    path_target = os.path.join(path, 'target')
    out.create_dir(path_target)
    path_result = os.path.join(path, 'result')
    out.create_dir(path_result)

    img_target = f'{path_target}/{model2}_{rep_type}_{seed_name}.pth'
    img_result = f'{path_result}/{model2}_{rep_type}_{seed_name}.pth'
    torch.save(targets, img_target)
    torch.save(results, img_result)

    print(f'=> Saved representations in {path}')
