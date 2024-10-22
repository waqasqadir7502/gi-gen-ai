import inquirer from "inquirer";

let toDo = ["Code", "Play Games", "Sleep", "Learning"];

let firstRes = await inquirer.prompt([
  {
    name: "question1",
    type: "list",
    message: "Welcome to your daily todolist!",
    choices: ["Add or Edit your todolist", "Show todolist"],
  },
]);

if (firstRes.question1 === "Show todolist") {
  console.log(toDo.join("\n"));
}

if (firstRes.question1 === "Add or Edit your todolist") {
  let secRes = await inquirer.prompt([
    {
      name: "question2",
      type: "list",
      message: "What would you like to do to your list",
      choices: ["Add items", "Remove items"],
    },
  ]);
  
  if (secRes.question2 === "Add items") {
    let condition = true;

    while (condition) {
      let thirdRes = await inquirer.prompt([
        {
          name: "question3",
          type: "input",
          message: "Add Items To Your List",
        },
        {
          name: "question4",
          type: "confirm",
          message: "Add More Items To Your List",
          default: "false",
        },
      ]);

      toDo.push(thirdRes.question3);
      condition = thirdRes.question4;
      console.log(toDo);
      if (thirdRes.question4 == "false") {
        break;
      };
    };
  }
  else if (secRes.question2 === "Remove items"){
    let fourthRes = await inquirer.prompt([
        {
            name: "question5",
            message:"Select the items to remove",
            type : "checkbox",
            choices : toDo, 
        }
    ]);
    
    toDo.filter(item => !fourthRes.question5.includes(item));
    
    // console.log(toDo.join("\n"))

  }
}
