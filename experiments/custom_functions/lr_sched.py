# LR SCHEDULER
def iwae_lr_sched(epoch):
    ''' based on 3280 epochs '''
    i_sum = 0
    i = 0
    i_sum = i_sum + 3**i 
    while epoch+1 > i_sum:
        i = i+1
        i_sum = i_sum + 3**i
    lr = .001*10**(-i/7.0)
    print(epoch, ': lr = ', lr)
    return lr

def alemi_300(epoch):
    if epoch <= 150:
        lr = .0003
    else:
        lr = 0.0003*(1-(300-epoch)/150)
    return lr

def alemi_400(epoch):
    if epoch <= 200:
        lr = .0003
    else:
        lr = 0.0003*(1-(400-epoch)/200)
    return lr


def alemi_800(epoch):
    if epoch <= 400:
        lr = .0003
    else:
        lr = 0.0003*(1-(800-epoch)/400)
    return lr

def alemi_lr_sched(epoch):
    if epoch <= 100:
        lr = .0003
    else:
        lr = .0003*(1-(200-epoch)/100)
    return lr

def alemi_0005(epoch):
    if epoch <= 100:
        lr = .0005
    else:
        lr = .0005*(1-(200-epoch)/100)
    return lr

def alemi_001(epoch):
    if epoch <= 100:
        lr = .001
    else:
        lr = .001*(1-(200-epoch)/100)
    return lr

def alemi_001_400(epoch):
    if epoch <= 200:
        lr = .001
    else:
        lr = .001*(1-(400-epoch)/200)
    return lr


# old version of learning rate schedule... 
# use something more general like this and pass kwargs?
def broken_elbo_mnist(epoch, base = 3*10**-4, constant = 100, total = 200):
    if epoch < constant:
        return base
    elif epoch < total:
        return (1 - (epoch - constant)/(total-constant))*base
    else:
        raise ValueError('Epoch outside anneal range of Alemi et al 2018 range. Modify "total" argument in anneal_fn')
