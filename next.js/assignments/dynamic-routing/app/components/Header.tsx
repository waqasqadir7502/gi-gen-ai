import Navlink from "./Navlink"

function Header(){
    return(
        <div>
            <nav className="flex flex-row justify-center h-20 items-center shadow-md">
            <ul className="flex gap-4 ">
                <Navlink href="/"  name="Home"/>
                <Navlink href="/country" name="Country"/>
            </ul>
            </nav>
        </div>
    )
}

export default Header