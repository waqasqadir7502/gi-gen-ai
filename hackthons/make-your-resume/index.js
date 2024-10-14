"use strict";
var _a, _b, _c, _d;
console.log("Connected!");
let uploadedImageData = null;
//Creating Resume
(_a = document.getElementById("resume-form")) === null || _a === void 0 ? void 0 : _a.addEventListener("submit", (ev) => {
    var _a, _b, _c;
    ev.preventDefault();
    // Calling form Inputs
    const fname = document.getElementById("f-name").value;
    const lname = document.getElementById("l-name").value;
    const email = document.getElementById("email").value;
    const contactNum = document.getElementById("contact-num").value;
    const eduSelect = document.getElementById("edu-select").value;
    const roleInfo = document.getElementById("role-info").value;
    const dateFrom = document.getElementById("datefrom").value;
    const dateTo = document.getElementById("dateto").value;
    const aboutInfo = document.getElementById("selfinfo").value;
    const name = fname + lname + Math.floor(Math.random() * 9999);
    const resumeInfo = {
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
    const fnamePara = document.getElementById("fname");
    const lnamePara = document.getElementById("lname");
    const emailPara = document.getElementById("emailp");
    const contactPara = document.getElementById("contactnum");
    const eduPara = document.getElementById("edu");
    const skillInfo = document.getElementById("skill-display-list");
    const workPara = document.getElementById("workdisplay");
    const dateFromPara = document.getElementById("datefromdis");
    const dateToPara = document.getElementById("datetodis");
    const selfInfoPara = document.getElementById("tellus-info");
    const skillList = document.getElementById("skilllist");
    const userName = document.getElementById("username");
    if (fname &&
        lname &&
        email &&
        contactNum &&
        roleInfo &&
        dateFrom &&
        dateTo &&
        eduSelect &&
        skills &&
        aboutInfo &&
        name) {
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
            localStorage.setItem("profileImage", uploadedImageData);
        }
        //Empty Inputs after submit
        document.getElementById("f-name").value = '';
        document.getElementById("l-name").value = '';
        document.getElementById("email").value = '';
        document.getElementById("contact-num").value = '';
        document.getElementById("edu-select").value = '';
        document.getElementById("role-info").value = '';
        document.getElementById("datefrom").value = '';
        document.getElementById("dateto").value = '';
        document.getElementById("selfinfo").value = '';
        skillList.style.display = "none";
        document.getElementById("resume-form").style.display = 'none';
        document.getElementById("resume-page").style.display = "block";
    }
    else {
        console.error("Some Inputs might be missing please recheck!!");
    }
    ;
    //Download Functionality
    (_a = document.getElementById("downbtn")) === null || _a === void 0 ? void 0 : _a.addEventListener("click", () => {
        const resume = document.getElementById("resume-section");
        const options = {
            margin: 0,
            filename: `${(userName === null || userName === void 0 ? void 0 : userName.textContent) || "resume"}_resume.pdf`,
            image: { type: "jpeg", quality: 0.98 },
            html2canvas: { scale: 2, useCORS: true },
            jsPDF: { unit: "in", format: "letter", orientation: "portrait" }
        };
        html2pdf().from(resume).set(options).save();
    });
    // Creating Shareable Link
    (_b = document.getElementById("sharebtn")) === null || _b === void 0 ? void 0 : _b.addEventListener("click", () => {
        const fname = document.getElementById("f-name").value;
        const lname = document.getElementById("l-name").value;
        const email = document.getElementById("email").value;
        const contactNum = document.getElementById("contact-num").value;
        const eduSelect = document.getElementById("edu-select").value;
        const roleInfo = document.getElementById("role-info").value;
        const dateFrom = document.getElementById("datefrom").value;
        const dateTo = document.getElementById("dateto").value;
        const aboutInfo = document.getElementById("selfinfo").value;
        //converting skills array to JSON String to pass via URL
        const skillsParam = JSON.stringify(skills);
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
            uploadedImageData: uploadedImageData || "", // Only include if image exists
            skills: skillsParam // Pass skills as a JSON string
        }).toString();
        console.log(queryParams);
        const shareableLink = `${window.location.origin}?${queryParams}`;
        // navigator.clipboard.writeText(shareableLink);
        prompt("This is link", shareableLink);
        // alert('Link Copied!')
    });
    // Go back functionality
    (_c = document.getElementById("createnew")) === null || _c === void 0 ? void 0 : _c.addEventListener("click", () => {
        window.location.reload();
    });
    //     // To Load Data on load
    window.addEventListener("load", () => {
        const params = new URLSearchParams(window.location.search);
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
            skillsArray = JSON.parse(skillsParam);
        }
        ;
        // If all mandatory fields are availabe, load the data into DOM 
        if (fname && lname && email && contactNum && eduSelect && roleInfo && dateFrom && dateTo && aboutInfo) {
            document.getElementById("fname").textContent = fname;
            document.getElementById("lname").textContent = lname;
            document.getElementById("emailp").textContent = email;
            document.getElementById("contactnum").textContent = contactNum;
            document.getElementById("edu").textContent = eduSelect;
            document.getElementById("workdisplay").textContent = roleInfo;
            document.getElementById("datefromdis").textContent = dateFrom;
            document.getElementById("datetodis").textContent = dateTo;
            document.getElementById("tellus-info").textContent = aboutInfo;
            //For image
            if (uploadImage) {
                document.getElementById("profileImage").src = uploadImage;
            }
            //For skills
            if (skillsArray.lenght > 0) {
                const skillList = document.getElementById("skill-list-display");
                skillList.innerHTML = "";
                skillsArray.forEach((skill) => {
                    const listItem = document.createElement('li');
                    listItem.textContent = skill;
                    skillList.appendChild(listItem);
                });
            }
            document.getElementById("resume-page").style.display = "block";
            document.getElementById("resume-form").style.display = 'none';
        }
    });
});
// Creating Edit Functionality 
(_b = document.getElementById("editbtn")) === null || _b === void 0 ? void 0 : _b.addEventListener("click", () => {
    //Recalling inputs and replacing generated form fields
    document.getElementById("f-name").value = document.getElementById("fname").textContent || '';
    document.getElementById("l-name").value = document.getElementById("lname").textContent || '';
    document.getElementById("email").value = document.getElementById("emailp").textContent || '';
    document.getElementById("contact-num").value = document.getElementById("contactnum").textContent || '';
    document.getElementById("edu-select").value = document.getElementById("edu").textContent || '';
    document.getElementById("role-info").value = document.getElementById("workdisplay").textContent || '';
    document.getElementById("datefrom").value = document.getElementById("datefromdis").textContent || '';
    document.getElementById("dateto").value = document.getElementById("datetodis").textContent || '';
    document.getElementById("selfinfo").value = document.getElementById("tellus-info").textContent || '';
    const skillList = document.getElementById("skilllist");
    skillList.style.display = "block";
    document.getElementById("resume-form").style.display = 'block';
    document.getElementById("resume-page").style.display = 'none';
});
//Add Skill Functionality
let skills = [];
function updatedSkills() {
    const skillList = document.getElementById("skilllist");
    skillList.innerHTML = ""; //This will Clear the list before updating
    skills.forEach((skill, index) => {
        //To Create List Item Element
        const listItem = document.createElement("li");
        listItem.textContent = skill;
        //To Delete List Item Element
        const removeBtn = document.createElement("button");
        removeBtn.textContent = "X";
        removeBtn.className = "removebtn";
        removeBtn.addEventListener("click", () => {
            removeSkill(index);
        });
        listItem.appendChild(removeBtn);
        skillList.appendChild(listItem);
    });
}
function removeSkill(index) {
    skills.splice(index, 1);
    updatedSkills();
}
(_c = document.getElementById("addskillbtn")) === null || _c === void 0 ? void 0 : _c.addEventListener("click", (ev) => {
    ev.preventDefault();
    const skill = document.getElementById("addskill").value.trim();
    if (skill) {
        skills.push(skill);
        document.getElementById("addskill").required = false;
        updatedSkills();
        document.getElementById("addskill").value = ""; //This will Clear The input field after adding one skill
    }
    else {
        alert("Please Add Skill");
    }
});
// Image adding
(_d = document.getElementById("imageUpload")) === null || _d === void 0 ? void 0 : _d.addEventListener("change", (ev) => {
    var _a;
    const fileInput = ev.target;
    const file = (_a = fileInput.files) === null || _a === void 0 ? void 0 : _a[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            var _a, _b;
            const imgEle = document.getElementById("profileImage");
            imgEle.src = (_a = e.target) === null || _a === void 0 ? void 0 : _a.result;
            uploadedImageData = (_b = e.target) === null || _b === void 0 ? void 0 : _b.result;
        };
        reader.readAsDataURL(file);
    }
    else {
        console.log("Error Getting Image Data");
    }
});
