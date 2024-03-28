import inquirer from "inquirer";

const answer = await inquirer.prompt([
  { message: "Enter First Number", type: "number", name: "FirstNumber" },
  { message: "Enter Second Number", type: "number", name: "SecondNumber" },
  { message: "Select the Operator", type: "list", name: "operator" ,choices : ["Addition", "Subraction", "Division","Multiplication"]},
]);

if (answer.operator === "Addition"){
    console.log(answer.FirstNumber + answer.SecondNumber);
}
else if(answer.operator === "Subraction") {
    console.log(answer.FirstNumber - answer.SecondNumber);
}else if(answer.operator === "Division") {
    console.log(answer.FirstNumber / answer.SecondNumber);
}else if(answer.operator === "Multiplication") {
    console.log(answer.FirstNumber * answer.SecondNumber);
}else{
    console.log("Please Enter A Valid Operator!")
}