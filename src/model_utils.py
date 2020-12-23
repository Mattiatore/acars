import torch

from tqdm import tqdm
from pathlib import Path
from numpy import mean

from src.data_utils import one_hot_encode, get_alphabet_dict

from logging import getLogger
from logging.config import fileConfig

logconfig = Path('.') / 'logging_config.ini'
fileConfig(logconfig)
logger = getLogger()


def train(model, train_loader, optimiser, criterion, num_epochs, print_every=500):
    """
    Training procedure

    :param model: torch.nn.Module: Model to train
    :param train_loader: torch.DataLoader: DataLoader with training dataset
    :param optimiser: torch.optim.Module: Optimiser object
    :param criterion: torch.nn.loss: Loss function
    :param num_epochs: int: Number of training epochs
    :param print_every: int: Logging config
    """
    model.train()

    if torch.cuda.is_available():
        logger.info('Run on available GPU')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    for epoch in range(num_epochs):

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_num, batch in progress_bar:
            inputs, labels = batch

            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimiser.zero_grad()
            logits = model(inputs)

            loss = criterion(logits, labels.long())
            loss.backward()

            optimiser.step()

            if (batch_num % print_every) == 0:
                logger.debug('Epoch [%d/%d], Batch[%d/%d], Loss: %.4f' %(epoch+1, num_epochs, batch_num, len(train_loader), loss.item()))

    logger.info('Finished training with final training loss: %.4f'%(loss.item()))


def evaluate(model, test_loader, criterion):
    """
    Evaluation procedure

    :param model: torch.nn.Module: A trained model
    :param test_loader: torch.DataLoader: DataLoader with evaluation dataset
    :param criterion: torch.nn.loss: Loss function

    :return: validation: dict: Dictinary with evaluation metrics across batches
    """
    model.eval()

    if torch.cuda.is_available():
        logger.info('Run on available GPU')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    validation = {'accuracy': [],
                  'avg_loss': [],
                  'label': [],
                  'predicted': []}

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))

    for batch_num, batch in progress_bar:
        inputs, labels = batch

        if torch.cuda.is_available():
            inputs = inputs.to(device)
            labels = labels.to(device)

        with torch.no_grad():
            logits = model(inputs)
            predicted = get_labels(logits)

        validation['label'].append(labels.cpu().detach().numpy().flatten())
        validation['predicted'].append(predicted.cpu().detach().numpy().flatten())

        acc = get_accuracy(logits, labels).cpu().detach().numpy()
        validation['accuracy'].extend(list(acc.flatten()))

        loss = criterion(logits, labels.long())
        avg_loss = torch.mean(loss.data).cpu().detach().numpy()
        validation['avg_loss'].extend(list(avg_loss.flatten()))

    logger.info('Finished evaluation with average accuracy %.4f and average loss %.4f' %(mean(validation['accuracy']), mean(validation['avg_loss'])))

    return validation


def predict(df, model, criterion, alphabet, config):
    """
    Get predictions from trained model. Runs in batches to avoid overloading memory.

    :param df: pd.DataFrame: Test data
    :param model: torch.nn.Module: A trained model
    :param criterion: torch.nn.loss: Loss function
    :param alphabet: str: Full alphabet
    :param config: dict: Model configuration

    :return: df: DataFrame: Test data with predicted labels added
    :return: validation: dict: Dictinary with evaluation metrics across batches
    """
    logger.info('Start prediction on %d test samples'%(len(df)))

    if torch.cuda.is_available():
        logger.info('Run on available GPU')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    alphabet_dict = get_alphabet_dict(alphabet)

    # Tokenise messages
    messages = list(df['Txt'].values)
    inputs = one_hot_encode(messages, alphabet_dict, config['max_seq_length'])
    inputs = torch.from_numpy(inputs)
    labels = df['Label'].apply(lambda x: config['labels'][x]).values
    labels = torch.from_numpy(labels)

    # Get predictions
    model.eval()
    model.to(device)

    df['Prediction'] = None

    validation = {'accuracy': [],
                  'avg_loss': []}

    batch_size = config['batch_size']
    num_batches = len(inputs) // batch_size

    for i in tqdm(range(num_batches)):

        batch_x = inputs[i*batch_size:(i+1)*batch_size]
        batch_y = labels[i*batch_size:(i+1)*batch_size]

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        predicted = get_labels(logits)
        predicted = predicted.cpu().detach().numpy()

        df['Prediction'].iloc[i*batch_size:(i+1)*batch_size] = predicted

        acc = get_accuracy(logits, batch_y).cpu().detach().numpy()
        validation['accuracy'].extend(list(acc.flatten()))

        loss = criterion(logits, batch_y.long())
        avg_loss = torch.mean(loss.data).cpu().detach().numpy()
        validation['avg_loss'].extend(list(avg_loss.flatten()))

    if len(inputs) % batch_size > 0:
        batch_x = inputs[num_batches*batch_size:]
        batch_y = labels[num_batches*batch_size:]

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        predicted = get_labels(logits)
        predicted = predicted.cpu().detach().numpy()

        df['Prediction'].iloc[num_batches*batch_size:] = predicted

        acc = get_accuracy(logits, batch_y).cpu().detach().numpy()
        validation['accuracy'].extend(list(acc.flatten()))

        loss = criterion(logits, batch_y.long())
        avg_loss = torch.mean(loss.data).cpu().detach().numpy()
        validation['avg_loss'].extend(list(avg_loss.flatten()))

    logger.info('Finished evaluation with average accuracy: %.4f'%mean(validation['accuracy']))
    inverse_labels = {v:k for k,v in config['labels'].items()}
    df['Prediction'] = df['Prediction'].apply(lambda x: inverse_labels[x])

    return df, validation


def get_accuracy(logits, labels):
    """ Calculate classification accuracy. """
    predicted = get_labels(logits)
    correct = predicted.eq(labels)

    return correct.sum().float() / correct.nelement()


def get_labels(logits):
    """ Get class labels from predicted logits. """
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    labels = torch.argmax(probabilities, 1)

    return labels
