import Link from "next/link"
import Image from "next/image"
import "./footer.css"

export default function Footer (){

return(

<div>
    <footer >
        <div className="footer-top flex justify-center items-center">
            <div className="footer-grid">
        <div className="footer-col-1 ">
            <h2>Hekto</h2>
            <div className="footer-newsletter">
                <input type="text" placeholder="Enter Email Address" />
                <button className="newsletter-btn">Sign up</button>
            </div>
            <p>Contact Info <br />
            17 Princess Road, London, Greater London NW1 8JR, UK</p>
        </div>
        <div className="footer-col-2">
            <h2>Categories</h2>
            <ul>
                <li><Link href="#">Laptops & Computers</Link></li>
                <li><Link href="#">Cameras & Photography</Link></li>
                <li><Link href="#">Smart Phones & Tablets</Link></li>
                <li><Link href="#">Video Games & Consoles</Link></li>
                <li><Link href="#">Waterproof Headphones</Link></li>
            </ul>
        </div>
        <div className="footer-col-3">
        <h2>Customer Care</h2>
            <ul>
                <li><Link href="#">My Account</Link></li>
                <li><Link href="#">Discount</Link></li>
                <li><Link href="#">Returns</Link></li>
                <li><Link href="#">Orders History</Link></li>
                <li><Link href="#">Order Tracking</Link></li>
            </ul>
        </div>
        <div className="footer-col-4">
            <h2>Pages</h2>
            <ul>
                <li><Link href="#">Blog</Link></li>
                <li><Link href="#">Browse the Shop</Link></li>
                <li><Link href="#">Category</Link></li>
                <li><Link href="#">Pre-Built Pages</Link></li>
                <li><Link href="#">Visual Composer Elements</Link></li>
                <li><Link href="#">WooCommerce Pages</Link></li>
            </ul>
        </div>
        </div>
        </div>
        <div className="footer-bottom flex justify-center items-center">
            <div className="footer-bottom-inner flex justify-between">
            <div className="footer-para">
                <p>©Webecy - All Rights Reserved</p>
            </div>
            <div className="footer-icons">
                <Image src="/icons/footer-social-icons.png" alt="socialicons" width={80} height={20} />
            </div>
            </div>
        </div>
    </footer>
</div>
)
}