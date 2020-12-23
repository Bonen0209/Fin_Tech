import argparse
import torch
import torchvision
from tqdm import tqdm
import data_loader.data_loaders as module_data
import trainer.loss as module_loss
import trainer.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        transform_args=config['data_loader']['args']['transform_args'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=8
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):

            #####################################################################################
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook

            data, target = data.to(device), target.to(device)

            #model.conv1.register_forward_hook(get_activation('conv2'))
            model.model.features[1].register_forward_hook(get_activation('feature1'))
            output = model(data)

            #torchvision.utils.save_image(data[0].cpu(), 'CNN_original.png', normalize=True)
            torchvision.utils.save_image(data[0].cpu(), 'Alexnet_original.png', normalize=True)
            #torchvision.utils.save_image(activation['conv2'][0].unsqueeze(1), 'CNN_activation.png', normalize=True)
            torchvision.utils.save_image(activation['feature1'][0].unsqueeze(1), 'Alexnet_activation.png', normalize=True)

            #images = torch.zeros_like(data).repeat(1, 3, 1, 1)
            images = torch.zeros((data.size(0), data.size(1), 28, 28))
            pred = torch.argmax(output, dim=1)

            for i, img in enumerate(data):
                #image = img.cpu().repeat(3, 1, 1)
                image = img.cpu()
                image = torchvision.transforms.ToPILImage()(image)
                image = torchvision.transforms.Resize(28)(image)

                dict_label = {
                    0: 'T-shirt/top',
                    1: 'Trouser',
                    2: 'Pullover',
                    3: 'Dress',
                    4: 'Coat',
                    5: 'Sandal',
                    6: 'Shirt',
                    7: 'Sneaker',
                    8: 'Bag',
                    9: 'Ankle boot'
                }

                from PIL import ImageDraw
                if pred[i] == target[i]:
                    ImageDraw.Draw(image).text((0, 0), dict_label[pred[i].item()], fill='rgb(0, 255, 0)')
                else:
                    ImageDraw.Draw(image).text((0, 0), dict_label[pred[i].item()], fill='rgb(255, 0, 0)')

                image = torchvision.transforms.ToTensor()(image)
                images[i] = image

            #torchvision.utils.save_image(images, 'CNN_predict.png')
            torchvision.utils.save_image(images, 'Alexnet_predict.png')
                
            #####################################################################################

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
