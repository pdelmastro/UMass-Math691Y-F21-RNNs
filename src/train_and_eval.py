"""
Methods for training and evaluating networks, specific to this experiment

Branched from dynamics_and_learning.src.train_and_eval on 8/25
"""
import numpy as np
import torch
import torch.nn as nn

def train(network, u, y, n_epochs, batch_size, lr, lr_decay=1.0,
        weight_decay=0.0, clip_gradient_thres=None, batches_per_eval=1,
        batches_til_first_lr_update=512, verbose=False, test_set=None,
        save_intermediate_params=False
    ):
    """
    Trains the provided 'network' to solve the task (u,y).
    Training is performed using ADAM optimizer with default beta settings.

    ARGUMENTS

        network             :   torch.nn.Module to be trained

        u                   :   Input sequences

        y                   :   Target output sequences

        n_epochs            :   Number of passes through the training set
                                Default: 1

        batch_size          :   Number of training examples per batch

        lr                  :   Initial learning rate

        lr_decay            :   Exponential decay factor for learning rate
                                Default: 1 (i.e. no decay)

        weight_decay        :   L_2 regularization
                                Default: 0.0

        clip_gradient_thres :   Gradients clipping threshold, under L_inf norm.
                                Set to 'None' for no clipping
                                Default: None

        batches_til_first_lr_update
                            :   learning rate decays by the factor
                                'lr decay' after
                                    'batches_til_first_lr_update' batches
                                    '2 * batches_til_first_lr_update' batches
                                    '4 * batches_til_first_lr_update' batches
                                    ...
                                    '2^n * batches_til_first_lr_update' batches

        verbose             :   Boolean. Set to True to print out updates during
                                training
                                Default: False

        test_set            :   Tuple (u_test,y_test) or None, storing
                                    dataset used for evaluating the network.
                                If set to 'None', this method uses the train
                                    set for evaluation.
                                Default: None

        save_intermediate_params  :   Flag. Set to True for this function to
                                return the sequence of parameters the network
                                goes through during the training process
    """
    # Useful constates
    T = u.shape[1]
    N_out = network.N_out
    n_batches = int(u.shape[0] / batch_size)

    # Loss function and batch data shapes
    mse_loss = nn.MSELoss()
    output_shape = (batch_size * T, -1)

    # Optimizers
    optimizer = torch.optim.Adam(
        network.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer, gamma=lr_decay
                )
    n_lr_updates = 0
    batches_til_lr_update = batches_til_first_lr_update

    # Test Set
    if test_set is None:
        u_test, y_test = (u, y)
    else:
        u_test, y_test = test_set

    # Arrays to store losses and accuracy
    n_evals = int(n_batches / batches_per_eval)
    losses = np.zeros((n_epochs, n_evals))

    # List to store the sequence of network parameters observed during training
    if save_intermediate_params:
        intermediate_params = []

    # opt. loops
    finished = False
    for ep in range(n_epochs):
        for b in range(n_batches):
            # Save the intermediate parameter?
            if save_intermediate_params:
                intermediate_params.append(network.numpy_state_dict())

            # Reset the gradients
            optimizer.zero_grad()

            # Network predictions
            rnn_out = network(u[batch_size * b : batch_size * (b+1)])
            if isinstance(rnn_out, tuple):
                y_pred = rnn_out[-1]
            else:
                y_pred = rnn_out

            # Compute loss
            loss = mse_loss(
                y_pred.view(batch_size * T, -1),
                y[batch_size * b : batch_size * (b+1)].view(output_shape)
            )

            # Compute the gradients of the loss wrt network parameters
            loss.backward()

            # Clip gradients for parameters
            if clip_gradient_thres is not None:
                nn.utils.clip_grad_norm_(
                    network.parameters(), clip_gradient_thres
                )

            # Update weights
            optimizer.step()

            # Network evaluation
            if (b % batches_per_eval) == (batches_per_eval-1):
                # Evaluation
                ev = int(b / batches_per_eval)
                losses[ep,ev] = evaluate(network, u_test, y_test)
                # Progress update
                if verbose:
                    fmt = "  Epoch {}/{}, Eval {}/{}, Loss: {:.3f}"
                    print(
                        fmt.format(
                            ep+1, n_epochs, ev+1, n_evals, losses[ep,ev]
                        )
                    )
                # Break if loss is practically 0
                if losses[ep,ev] < 5e-4:
                    # Fill in the rest of this epoch
                    losses[ep,ev:] = losses[ep,ev]
                    # Fill in future epochs
                    for ep_f in range(ep, n_epochs):
                        losses[ep_f,:] = losses[ep,ev]
                    finished=True
                    break

            if finished:
                break

            # Update the learning rate
            if batches_til_lr_update == 0:
                lr_scheduler.step()
                n_lr_updates += 1
                batches_til_lr_update = batches_til_first_lr_update * (2 ** n_lr_updates)
            else:
                batches_til_lr_update -= 1

    # Save the final state of the model
    if save_intermediate_params:
        intermediate_params.append(network.numpy_state_dict())

    # Return the losses (and intermediate params if the flag is set)
    to_return = [losses.reshape(n_epochs * n_evals)]
    if save_intermediate_params:
        to_return.append(intermediate_params)
    return to_return



def evaluate(network, u, y):
    """
    ARGUMENTS

        network             :   torch.nn.Module to be evaluated

        u                   :   Input sequences

        y                   :   Target output sequences

    RETURNS

        loss                :   MSE loss of the network on the mapping u -> y

    """
    # Output loss function and shape
    mse_loss = nn.MSELoss()
    output_shape = (y.shape[0] * y.shape[1], -1)

    with torch.no_grad():
        # Get the network predictions on the input sequence
        rnn_out = network(u)
        if isinstance(rnn_out, tuple):
            m_pred, y_pred = rnn_out
        else:
            y_pred = rnn_out

        # Compute the loss of the network
        loss = mse_loss(y_pred, y)

    # Return
    return loss
