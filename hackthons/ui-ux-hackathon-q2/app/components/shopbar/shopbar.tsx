import Image from "next/image";
// import Link from "next/link";
import "./shopbar.css"

export default function ShopBar():any{
    return(
    <div className="top-shop-bar flex justify-center align-center ">
        <div className="top-shop-inner flex justify-between items-center">
    <div className="shop-col-1">
      <h3>Ecommerce Acceories & Fashion item </h3>
      <p>About 9,620 results (0.62 seconds)</p>
    </div>
    <div className="shop-col-2 flex gap-x-5">
      <div className="page-per flex">
        <p>Page Per:</p>
        <input type="text"/>
      </div>
      <div className="sort-by flex gap-x-5">
      <p>Sort By:</p>
      <select name="" id="">
        <option value="">Best Match</option>
      </select>
      </div>
      <div className="view flex gap-x-5">
      <p>View</p>
      <button><Image src="./icons/shop-grid.svg" alt="shop grid view" width={12} height={12}/></button>
      <button><Image src="./icons/shop-list.svg" alt="shop grid view" width={12} height={12}/></button>
      <input type="text"/>
      </div>
    </div>
    </div>
   </div>
    )
}