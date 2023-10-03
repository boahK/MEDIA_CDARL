'''create dataset and dataloader'''
import logging
import torch.utils.data

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        # raise NotImplementedError('Dataloader [{:s}] is not found.'.format(phase))

def create_dataset_xcad(dataset_opt, phase):
    '''create dataset'''
    from data.XCAD_dataset import XCADDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'], split=phase)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset