use anyhow::Result;
use structopt::StructOpt;
use tch::{kind, nn, nn::Module, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

const IMAGE_DIM: i64 = 784;
const LABELS: i64 = 10;

#[derive(Debug)]
struct ConvNet {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ConvNet {
    fn new(vs: &nn::Path) -> ConvNet {
        let conf: nn::ConvConfig = Default::default();
        println!("Config: {:?}", conf);
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 1024, Default::default());
        let fc2 = nn::linear(vs, 1024, 10, Default::default());
        ConvNet {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::ModuleT for ConvNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .dropout(0.5, train)
            .apply(&self.fc2)
    }
}

fn accuracy(t1: &Tensor, t2: &Tensor) -> f64 {
    if t1.dim() != 1 || t2.dim() != 1 {
        panic!("");
    }
    if t1.size1().unwrap() != t2.size1().unwrap() {
        panic!("");
    }
    let sz = t1.size1().unwrap();
    let mut equal: f64 = 0.0;
    for i in 0..sz {
        if t1.get(i) == t2.get(i) {
            equal += 1.0;
        }
    }
    equal / sz as f64
}

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    #[structopt(short = "m", long = "mode")]
    mode: String,
}

fn main() -> Result<()> {
    let options = Opt::from_args();
    let m = tch::vision::mnist::load_dir("data")?;
    if options.mode == "linear" {
        let train_images = m.train_images;
        println!("{}", train_images.dim());
        for sz in train_images.size() {
            println!("Shape: {}", sz);
        }
        let image = train_images.get(0);
        println!("One image shape: {}", image.size()[0]);
        for i in 0..image.size1().unwrap() {
            print!("{},", image.double_value(&[i]));
        }
        println!("");

        let mut ws = Tensor::zeros(&[IMAGE_DIM, LABELS], kind::FLOAT_CPU).set_requires_grad(true);
        let mut bs = Tensor::zeros(&[LABELS], kind::FLOAT_CPU).set_requires_grad(true);

        for _epoch in 0..200 {
            let logits = train_images.mm(&ws) + &bs;
            let loss = logits
                .log_softmax(-1, kind::FLOAT_CPU.0)
                .nll_loss(&m.train_labels);
            println!("Loss: {:?}", loss);
            ws.zero_grad();
            bs.zero_grad();
            loss.backward();
            tch::no_grad(|| {
                ws += ws.grad() * (-1);
                bs += bs.grad() * (-1);
            });
            tch::no_grad(|| {
                let batch = m.test_images.slice(0, 0, 124, 1);
                let logits = batch.mm(&ws) + &bs;
                println!("Final logits: ");
                // logits.print();
                let predicts = logits.max_dim(1, false).1;
                println!(
                    "Acc: {}",
                    accuracy(&predicts, &m.test_labels.slice(0, 0, 124, 1))
                );
            })
        }
        //println!("Expected labels:");
        // m.test_labels.slice(0, 0, 124, 1).print();
        let logits = m.test_images.mm(&ws) + &bs;
        let predicts = logits.max_dim(1, false).1;
        println!("Total accuracy: {}", accuracy(&predicts, &m.test_labels));
        Ok(())
    } else if options.mode == "dense" {
        // sequential torch network
        // simple dense net
        const HIDDEN_NODES: i64 = 128;
        let vs = nn::VarStore::new(Device::Cpu);
        let net = nn::seq()
            .add(nn::linear(
                &vs.root() / "layer1",
                IMAGE_DIM,
                HIDDEN_NODES,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(
                &vs.root(),
                HIDDEN_NODES,
                LABELS,
                Default::default(),
            ));
        let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
        for epoch in 0..200 {
            let loss = net
                .forward(&m.train_images)
                .cross_entropy_for_logits(&m.train_labels);
            opt.backward_step(&loss);
            let test_accuracy = net
                .forward(&m.test_images)
                .accuracy_for_logits(&m.test_labels);
            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                f64::from(&loss),
                100. * f64::from(&test_accuracy),
            );
        }
        Ok(())
    } else if options.mode == "conv" {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let net = ConvNet::new(&vs.root());
        let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
        for epoch in 0..100 {
            for (bimages, blabels) in m.train_iter(256).shuffle().to_device(vs.device()) {
                let loss = net
                    .forward_t(&bimages, true)
                    .cross_entropy_for_logits(&blabels);
                opt.backward_step(&loss);
            }
            let test_accuracy =
                net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 1024);
            println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
        }
        Ok(())
    } else {
        panic!("Unexpected mode: {}", options.mode);
    }
}
