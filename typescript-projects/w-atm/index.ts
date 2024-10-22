#! /usr/bin/env node
import inquirer from "inquirer";

let myAmount = 10000;
let myPin = 1122;

let answer = await inquirer.prompt([
  {
    name: "pin",
    message: "Enter Your Pin Code XXXX",
    type: "number",
  },
]);

if (myPin === answer.pin) {
  let options = await inquirer.prompt([
    {
      name: "option",
      message: "Select Transaction Option.",
      type: "list",
      choices: ["Withdraw", "Check Balance", "Fast withdraw"],
    },
  ]);

  if (options.option === "Withdraw") {
    let addAmount = await inquirer.prompt([
      {
        name: "amount",
        message: "Enter The Desire Amount",
        type: "number",
      },
    ]);
    if (myAmount >= addAmount.amount) {
      myAmount -= addAmount.amount;
      console.log(`Your Remaining Amount is ${myAmount}`);
    } else {
      console.log(
        "Unable to Process the Transaction Due To Unsufficent Balance"
      );
    }
  } else if (options.option === "Check Balance") {
    console.log(myAmount);
  } else if (options.option === "Fast withdraw") {
    let fastCash = await inquirer.prompt([
      {
        name: "fastcash",
        message: "Please Select the Required Amount",
        type: "list",
        choices: ["1000", "2000", "5000", "10000", "15000", "20000"],
      },
    ]);
    if (myAmount >= fastCash.fastcash) {
      myAmount -= fastCash.fastcash;
      console.log(`Your Remaining Balance is now: ${myAmount} `);
    } else {
        console.log("Unable to Process the Transaction Due To Unsufficent Balance");
      }
      
  } else{
    console.log("Incorrect Pin Code! Please Try Again.")
  }
}