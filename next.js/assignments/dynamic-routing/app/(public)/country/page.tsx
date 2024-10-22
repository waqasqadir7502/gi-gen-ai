import Navlink from "@/app/components/Navlink"

function Country(){
    return(
        <div className="m-40 text-xl">
        <h1 className="text-center m-4">Featured Countries</h1>
        <div className="flex flex-row justify-center items-center list-none ">
        <Navlink href="/country/pakistan" name="Pakistan" />
        <Navlink href="/country/usa" name="USA"/>
        <Navlink href="/country/korea" name="Korea"/>
        <Navlink href="/country/china" name="China"/>
        <Navlink href="/country/morocco" name="Morocco"/>
        <Navlink href="/country/england" name="England"/>
        </div>
        </div>
    )
}

export default Country