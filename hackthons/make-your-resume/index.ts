console.log("Connected!");
declare const html2pdf: any;
let uploadedImageData: string | null = null;


//Creating Resume
document.getElementById("resume-form")?.addEventListener("submit", (ev) => {
    ev.preventDefault();

    // Calling form Inputs
    const fname = (document.getElementById("f-name") as HTMLInputElement).value;
    const lname = (document.getElementById("l-name") as HTMLInputElement).value;
    const email = (document.getElementById("email") as HTMLInputElement).value;
    const contactNum = (document.getElementById("contact-num") as HTMLInputElement).value;
    const eduSelect = (document.getElementById("edu-select") as HTMLSelectElement).value;
    const roleInfo = (document.getElementById("role-info") as HTMLInputElement).value;
    const dateFrom = (document.getElementById("datefrom") as HTMLInputElement).value;
    const dateTo = (document.getElementById("dateto") as HTMLInputElement).value;
    const aboutInfo = (document.getElementById("selfinfo") as HTMLTextAreaElement).value;
    const name = fname + lname + Math.floor(Math.random() * 9999);


    const resumeInfo: any =
    {
        fname,
        lname,
        email,
        contactNum,
        roleInfo,
        aboutInfo,
        dateFrom,
        dateTo,
        eduSelect,
        skills,
        name,
        uploadedImageData
    };

    //Storing Resume Data in Local Storage
    localStorage.setItem("resumeData", JSON.stringify(resumeInfo));

    // Calling References Display Div
    const fnamePara = document.getElementById("fname") as HTMLParagraphElement;
    const lnamePara = document.getElementById("lname") as HTMLParagraphElement;
    const emailPara = document.getElementById("emailp") as HTMLParagraphElement;
    const contactPara = document.getElementById("contactnum") as HTMLParagraphElement;
    const eduPara = document.getElementById("edu") as HTMLParagraphElement;
    const skillInfo = document.getElementById("skill-display-list") as HTMLOListElement;
    const workPara = document.getElementById("workdisplay") as HTMLParagraphElement;
    const dateFromPara = document.getElementById("datefromdis") as HTMLParagraphElement;
    const dateToPara = document.getElementById("datetodis") as HTMLParagraphElement;
    const selfInfoPara = document.getElementById("tellus-info") as HTMLParagraphElement;
    const skillList = document.getElementById("skilllist") as HTMLUListElement;
    const userName = document.getElementById("username") as HTMLParagraphElement;

    if (
        fname &&
        lname &&
        email &&
        contactNum &&
        roleInfo &&
        dateFrom &&
        dateTo &&
        eduSelect &&
        skills &&
        aboutInfo &&
        name
    ) {
        console.log(resumeInfo);
        fnamePara.textContent = fname;
        lnamePara.textContent = lname;
        emailPara.textContent = email;
        contactPara.textContent = contactNum;
        selfInfoPara.innerHTML = aboutInfo;
        eduPara.textContent = eduSelect;
        skillInfo.innerHTML = "";
        skills.forEach((skill) => {
            const listItem = document.createElement("li");
            listItem.textContent = skill;
            skillInfo.appendChild(listItem);
        });
        workPara.textContent = roleInfo;
        dateFromPara.innerHTML = dateFrom;
        dateToPara.innerHTML = dateTo;
        userName.innerHTML = name;

        if (uploadedImageData) {
            localStorage.setItem("profileImage", uploadedImageData)
        }

        //Empty Inputs after submit
        (document.getElementById("f-name") as HTMLInputElement).value = '';
        (document.getElementById("l-name") as HTMLInputElement).value = '';
        (document.getElementById("email") as HTMLInputElement).value = '';
        (document.getElementById("contact-num") as HTMLInputElement).value = '';
        (document.getElementById("edu-select") as HTMLSelectElement).value = '';
        (document.getElementById("role-info") as HTMLInputElement).value = '';
        (document.getElementById("datefrom") as HTMLInputElement).value = '';
        (document.getElementById("dateto") as HTMLInputElement).value = '';
        (document.getElementById("selfinfo") as HTMLTextAreaElement).value = '';
        skillList.style.display = "none";


        (document.getElementById("resume-form") as HTMLElement).style.display = 'none';
        (document.getElementById("resume-page") as HTMLElement).style.display =  "block";

    } else {
        console.error("Some Inputs might be missing please recheck!!");
    };



    //Download Functionality
    document.getElementById("downbtn")?.addEventListener("click", () => {

        const resume = document.getElementById("resume-section");
        const options = {
            margin: 0,
            filename: `${userName?.textContent || "resume"}_resume.pdf`,
            image: { type: "jpeg", quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true },
            jsPDF: { unit: "in", format: "letter", orientation: "portrait" }
        };

        html2pdf().from(resume).set(options).save();
    });


   // Creating Shareable Link
    document.getElementById("sharebtn")?.addEventListener("click", () => {
        const fname = (document.getElementById("f-name") as HTMLInputElement).value;
        const lname = (document.getElementById("l-name") as HTMLInputElement).value;
        const email = (document.getElementById("email") as HTMLInputElement).value;
        const contactNum = (document.getElementById("contact-num") as HTMLInputElement).value;
        const eduSelect = (document.getElementById("edu-select") as HTMLSelectElement).value;
        const roleInfo = (document.getElementById("role-info") as HTMLInputElement).value;
        const dateFrom = (document.getElementById("datefrom") as HTMLInputElement).value;
        const dateTo = (document.getElementById("dateto") as HTMLInputElement).value;
        const aboutInfo = (document.getElementById("selfinfo") as HTMLTextAreaElement).value;

        //converting skills array to JSON String to pass via URL
        const skillsParam = JSON.stringify(skills)

        const queryParams = new URLSearchParams({
            fname,
            lname,
            email,
            contactNum,
            eduSelect,
            roleInfo,
            dateFrom,
            dateTo,
            aboutInfo,
            uploadedImageData: uploadedImageData || "",  // Only include if image exists
            skills: skillsParam  // Pass skills as a JSON string
        }).toString();

        console.log(queryParams)

        const shareableLink = `${window.location.origin}?${queryParams}`;
        // navigator.clipboard.writeText(shareableLink);
        prompt("This is link", shareableLink)
        // alert('Link Copied!')
    });

// Go back functionality
document.getElementById("createnew")?.addEventListener("click", () => {
    window.location.reload()
})

//     // To Load Data on load

window.addEventListener("load", () => {
    const params = new URLSearchParams(window.location.search)

    //Fetching Data from URL Parameters

    const fname = params.get("fname");
    const lname = params.get("lname");
    const email = params.get("email");
    const contactNum = params.get("contactNum");
    const eduSelect = params.get("eduSelect");
    const roleInfo = params.get("roleInfo");
    const dateFrom = params.get("dateFrom");
    const dateTo = params.get("dateTo");
    const aboutInfo = params.get("aboutInfo");
    const uploadImage = params.get("uploadImageData");
    const skillsParam = params.get("skills");

    //Parsing Skills 
    let skillsArray = [];
    if (skillsParam) {
        skillsArray = JSON.parse(skillsParam)
    };

    // If all mandatory fields are availabe, load the data into DOM 
    if (fname && lname && email && contactNum && eduSelect && roleInfo && dateFrom && dateTo && aboutInfo) {
        (document.getElementById("fname") as HTMLParagraphElement).textContent = fname;
        (document.getElementById("lname") as HTMLParagraphElement).textContent = lname;
        (document.getElementById("emailp") as HTMLParagraphElement).textContent = email;
        (document.getElementById("contactnum") as HTMLParagraphElement).textContent = contactNum;
        (document.getElementById("edu") as HTMLParagraphElement).textContent = eduSelect;
        (document.getElementById("workdisplay") as HTMLParagraphElement).textContent = roleInfo;
        (document.getElementById("datefromdis") as HTMLParagraphElement).textContent = dateFrom;
        (document.getElementById("datetodis") as HTMLParagraphElement).textContent = dateTo;
        (document.getElementById("tellus-info") as HTMLParagraphElement).textContent = aboutInfo;

        //For image
        if (uploadImage) {
            (document.getElementById("profileImage") as HTMLImageElement).src = uploadImage;
        }

        //For skills
        if (skillsArray.lenght > 0) {
            const skillList = document.getElementById("skill-list-display") as HTMLOListElement
            skillList.innerHTML = "";
            skillsArray.forEach((skill: string | null) => {
                const listItem = document.createElement('li')
                listItem.textContent = skill
                skillList.appendChild(listItem)
            })
        }

        (document.getElementById("resume-page") as HTMLElement).style.display  = "block";
        (document.getElementById("resume-form") as HTMLElement).style.display = 'none';
    }
})



});
 

// Creating Edit Functionality 
document.getElementById("editbtn")?.addEventListener("click", () => {

    //Recalling inputs and replacing generated form fields
    (document.getElementById("f-name") as HTMLInputElement).value = (document.getElementById("fname") as HTMLParagraphElement).textContent || '';
    (document.getElementById("l-name") as HTMLInputElement).value = (document.getElementById("lname") as HTMLParagraphElement).textContent || '';
    (document.getElementById("email") as HTMLInputElement).value = (document.getElementById("emailp") as HTMLParagraphElement).textContent || '';
    (document.getElementById("contact-num") as HTMLInputElement).value = (document.getElementById("contactnum") as HTMLParagraphElement).textContent || '';
    (document.getElementById("edu-select") as HTMLSelectElement).value = (document.getElementById("edu") as HTMLParagraphElement).textContent || '';
    (document.getElementById("role-info") as HTMLInputElement).value = (document.getElementById("workdisplay") as HTMLParagraphElement).textContent || '';
    (document.getElementById("datefrom") as HTMLInputElement).value = (document.getElementById("datefromdis") as HTMLParagraphElement).textContent || '';
    (document.getElementById("dateto") as HTMLInputElement).value = (document.getElementById("datetodis") as HTMLParagraphElement).textContent || '';
    (document.getElementById("selfinfo") as HTMLTextAreaElement).value = (document.getElementById("tellus-info") as HTMLParagraphElement).textContent || '';
    const skillList = document.getElementById("skilllist") as HTMLUListElement;
    skillList.style.display = "block";

    (document.getElementById("resume-form") as HTMLElement).style.display = 'block';
    (document.getElementById("resume-page") as HTMLElement).style.display = 'none';


});

//Add Skill Functionality

let skills: string[] = [];

function updatedSkills() {
    const skillList = document.getElementById("skilllist") as HTMLUListElement;
    skillList.innerHTML = ""; //This will Clear the list before updating

    skills.forEach((skill, index) => {
        //To Create List Item Element
        const listItem = document.createElement("li");
        listItem.textContent = skill;

        //To Delete List Item Element
        const removeBtn = document.createElement("button");
        removeBtn.textContent = "X";
        removeBtn.className = "removebtn"
        removeBtn.addEventListener("click", () => {
            removeSkill(index);
        });

        listItem.appendChild(removeBtn);
        skillList.appendChild(listItem);
    });
}

function removeSkill(index: number) {
    skills.splice(index, 1);
    updatedSkills();
}

document.getElementById("addskillbtn")?.addEventListener("click", (ev) => {
    ev.preventDefault();
    const skill = (document.getElementById("addskill") as HTMLInputElement).value.trim();

    if (skill) {
        skills.push(skill);
        (document.getElementById("addskill") as HTMLInputElement).required = false;
        updatedSkills();
        (document.getElementById("addskill") as HTMLInputElement).value = ""; //This will Clear The input field after adding one skill
    } else {
        alert("Please Add Skill");
    }
});

// Image adding
document.getElementById("imageUpload")?.addEventListener("change", (ev) => {
    const fileInput = ev.target as HTMLInputElement;
    const file = fileInput.files?.[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const imgEle = document.getElementById("profileImage") as HTMLImageElement;
            imgEle.src = e.target?.result as string;
            uploadedImageData = e.target?.result as string;
        }
        reader.readAsDataURL(file);
    } else {
        console.log("Error Getting Image Data");
    }

})



