from .trainers.sick_trainer import SICKTrainer

class TrainerFactory(object):
    """
    Get the corresponding Trainer class for a particular dataset.
    """
    trainer_map = {
        'sick': SICKTrainer,
        'eng_fre':SICKTrainer
    }

    @staticmethod
    def get_trainer(dataset_name, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None, nce=False):
        if nce:
            trainer_map = TrainerFactory.trainer_map_nce
        else:
            trainer_map = TrainerFactory.trainer_map

        if dataset_name not in trainer_map:
            raise ValueError('{} is not implemented.'.format(dataset_name))

        return trainer_map[dataset_name](
            model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator
        )
