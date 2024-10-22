import inquirer from "inquirer";

const currency : any = {
  USD: 1, //Base currency
  EUR: 0.91,
  GBP: 0.76,
  INR: 74.57,
  PKR: 280,
};

let userRes = await inquirer.prompt([
  {
    name: "currencyselector1",
    message: "Pick your choice of currency!",
    type: "list",
    choices: ["USD", "EUR", "GBP", "INR", "PKR"],
  },
  {
    name: "currencyselector2",
    message: "Pick your choice of currency to convert!",
    type: "list",
    choices: ["USD", "EUR", "GBP", "INR", "PKR"],
  },
  {
    name: "amount",
    message: "Enter Your Amount!",
    type: "number",
  },
]);

let firstCurr = currency[userRes.currencyselector1]
let secCurr = currency[userRes.currencyselector2]
let amount = userRes.amount
let baseAmount = amount / firstCurr // USD based currency
let shownAmount = baseAmount * secCurr

console.log(shownAmount)
