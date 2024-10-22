import Link from "next/link"
import { NavlinkProp } from "../types/types"

const Navlink: React.FC<NavlinkProp> =({name, href})=>{
    return(
        <>
        <li >
        <Link href={href} className="hover:bg-black hover:text-white p-4 shadow-md">{name}</Link>
        </li>
        </>
    )
}

export default Navlink