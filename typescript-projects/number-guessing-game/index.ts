#! /usr/bin/env node

import inquirer from "inquirer";

const randomNumber = Math.floor(Math.random() * 10 + 1);
console.log(randomNumber)
const answer = await inquirer.prompt([{
    name :"userGuessedNumber",
    type : "number",
    message: "Guess The Number"
}])

if (answer.userGuessedNumber === randomNumber){
    console.log("You've guess the Correct Number!")
}else{
    console.log("You've Guessed The Wrong Number!")
}

// console.log(answer)