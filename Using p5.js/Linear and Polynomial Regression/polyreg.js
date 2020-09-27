// Initializing the empty lists

let x_vals=[]
let y_vals=[]


// initializing the learningrate and optimizer
let a, b,c ;

const learningRate= 0.01
const optimizer= tf.train.adam(learningRate);


// for p5 this function runs once when we reload the page or open it fot the first time
function setup() {
  // put setup code here
  createCanvas(1500,720); // creating a canvas with black background  
  background(0);

  a=tf.variable(tf.scalar(random(1)));  // initialising slope
  b=tf.variable(tf.scalar(random(1)));
  c=tf.variable(tf.scalar(random(1))); // initializing bias
}



//Loss function using mean squared error loss
function loss(pred,label){
  return pred.sub(label).square().mean()

}


//Predict funtion that takes an "array" and returns a tensor generally this function is reponsible for the memory leak 
function predict(xs){

  const x =tf.tensor1d(xs);
  const pred= x.square().mul(a).add(x.mul(b)).add(c)
  return pred;
}


//Funtion to take the points clicked 
function mousePressed(){

  //convert the image coordinate system to normalized coordinate system
  let x = map ( mouseX ,0,width, -1, 1); //map the distance between [0-width] -> [0,1]
  let y = map ( mouseY, 0,height, 1, -1);//map the distance between [0-height] -> [1,0]


  //pushing the coordinated to the globally declared x_vals and y_vals
  x_vals.push(x);
  y_vals.push(y);
  //console.log(`${mouseX}== ${x} , ${mouseY}== ${y}`)
}


//The draw funtion in p5 run infinitly  
function draw() {

    // To prevent the memory leak we enclose this in tidy
    tf.tidy(()=>{
      if(x_vals.length>0){
          const ys=tf.tensor1d(y_vals)

          optimizer.minimize(() => loss(predict(x_vals),ys))
        }
    })



     background(0);
     stroke(255);
     strokeWeight(4);

     //Loop to print the points that are clicked
    for (let i=0 ;i < x_vals.length; i++){

      // To plot the points we need to map the xs and ys back to the image coordinate system
      let px = map(x_vals[i],-1,1,0,width) 
      let py= map ( y_vals[i], -1 , 1, height, 0)
      point(px,py);
    }

    
   // We will plot the line 
   const curveX=[]
    for (let i=-1;i<1;i+=0.05)
    {
            curveX.push(i)
        }

    const ys= tf.tidy(()=> predict(curveX))
    let curveY = ys.dataSync() // in order to use the data using the index we have to use dataSync()
    ys.dispose() //disposeing ys in order to prevent memory leak

    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);

    for (let i=0; i<curveX.length; i++)
    {
        let x =map(curveX[i],-1,1,0,width)
        let y =map(curveY[i],-1, 1,height,0)

        vertex(x,y)

    }

    endShape();   



    //The values will be mapped according to the original coordinates


  // to check whether there is any memory leak  
  console.log(tf.memory().numTensors)


    
}