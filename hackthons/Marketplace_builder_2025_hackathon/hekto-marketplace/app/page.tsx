import Image from "next/image";
import "./home.css";
export default function Home() {
  return (
    <div>
      {/* Hero Section / Section 1 */}
      <div className="hero-section flex items-center justify-center ">
        <Image
          src="/homepage/hero-1.png"
          alt="light"
          width={387}
          height={387}
          className="hero-pic-1"
        />
        <Image
          src="/homepage/hero-2.png"
          alt="discount"
          width={660}
          height={660}
          className="hero-pic-2"
        />
        <Image
          src="/homepage/hero-3.png"
          alt="dot"
          width={15}
          height={15}
          className="hero-pic-3"
        />
        <div className="hero-inner-section">
          <p className="hero-p-1">Best Furniture For Your Castle....</p>
          <h1 className="hero-heading">
            New Furniture Collection <br />
            Trends in 2020
          </h1>
          <p className="hero-p-2">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Magna in
            est adipiscing <br />
            in phasellus non in justo.
          </p>
          <button className="shopnowbtn">Shop Now</button>
        </div>
        <Image
          src="/homepage/hero-page-nav.png"
          alt="Pageination"
          width={66}
          height={14}
          className="hero-pagi"
        />
      </div>

      {/* Featured Section / Section 2 */}
      <div className="featured-sec flex flex-col justify-center items-center">
        <h2>Featured Products</h2>
        <div className="feautred-pro grid grid-cols-4 gap-x-5">
          <div className="product">
            <div className="product-img flex items-center justify-center">
              <Image
                src="/homepage/featured1.png"
                width={175}
                height={175}
                alt="featured Chair"
              />
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4">
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
                <button className="color-btn-1"></button>
                <button className="color-btn-2"></button>
                <button className="color-btn-3"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>

          <div className="product second-prod ">
            <div className="product-img flex items-center justify-center ">
              <div className="producticons flex gap-x-3">
                <Image
                  src="/icons/featured-cart.svg"
                  alt="icon"
                  width={15}
                  height={15}
                />
                <Image
                  src="/icons/featured-heart.svg"
                  alt="icon"
                  width={15}
                  height={15}
                />
                <Image
                  src="/icons/featured-search.svg"
                  alt="icon"
                  width={15}
                  height={15}
                />
              </div>
              <button className="featured-vm-btn">View More</button>
              <Image
                src="/homepage/featured2.png"
                width={130}
                height={150}
                alt="featured Chair"
              />
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4 second-prod-details">
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
                <button className="color-btn-1"></button>
                <button className="color-btn-2"></button>
                <button className="color-btn-3 second-prod-btn"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>

          <div className="product ">
            <div className="product-img flex items-center justify-center">
              <Image
                src="/homepage/featured3.png"
                width={175}
                height={175}
                alt="featured Chair"
              />
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4">
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
                <button className="color-btn-1"></button>
                <button className="color-btn-2"></button>
                <button className="color-btn-3"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>

          <div className="product">
            <div className="product-img flex items-center justify-center">
              <Image
                src="/homepage/featured4.png"
                width={216}
                height={151}
                alt="featured Chair"
              />
            </div>
            <div className="product-details text-center flex flex-col gap-y-3 mt-4">
              <h3 className="featured-pro-name">Cantilever Chair</h3>
              <div className="flex justify-center items-center gap-x-1">
                <button className="color-btn-1"></button>
                <button className="color-btn-2"></button>
                <button className="color-btn-3"></button>
              </div>
              <p className="featured-para featured-pro-code">Code - Y523201</p>
              <p className=" featured-para featured-price">$42.00</p>
            </div>
          </div>
        </div>
        <div className="pagi-btns flex justify-center mt-10 gap-x-2">
          <button className="btn-1 "></button>
          <button className="btn-2 pagi-btn"></button>
          <button className="btn-3 pagi-btn"></button>
          <button className="btn-4 pagi-btn"></button>
        </div>
      </div>

      {/* Latest Section / Section 3 */}
      <div className="latest-sec flex flex-col justify-center items-center gap-y-6">
        <h2>Latest Products</h2>
        <div className="latest-option flex justify-center gap-10">
          <p className="arrival-para">New Arrival</p>
          <p>Best Seller</p>
          <p>Featured</p>
          <p>Special Offer</p>
        </div>
        <div className="latest-prod-grid grid grid-cols-3 gap-y-5">
          <div className="latest-prod">
            <div className="latest-prod-img flex justify-center items-center ">
              <Image
                src="/homepage/latest1.png"
                alt="latest product"
                width={223}
                height={229}
              />
            </div>
            <div className="latest-prod-details flex justify-between items-center">
              <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
              </div>
              <p className="latest-prod-price">
                $42.00 <span className="latest-discounted">$62.00</span>
              </p>
            </div>
          </div>

          <div className="latest-prod">
            <div className="latest-prod-img flex justify-center items-center ">
              <Image
                src="/homepage/latest2.png"
                alt="latest product"
                width={370}
                height={277}
              />
            </div>
            <div className="latest-prod-details flex justify-between items-center">
              <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
              </div>
              <p className="latest-prod-price">
                $42.00 <span className="latest-discounted">$62.00</span>
              </p>
            </div>
          </div>

          <div className="latest-prod">
            <div className="latest-prod-img flex justify-center items-center ">
              <Image
                src="/homepage/latest3.png"
                alt="latest product"
                width={222}
                height={222}
              />
            </div>
            <div className="latest-prod-details flex justify-between items-center">
              <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
              </div>
              <p className="latest-prod-price">
                $42.00 <span className="latest-discounted">$62.00</span>
              </p>
            </div>
          </div>

          <div className="latest-prod">
            <div className="latest-prod-img flex justify-center items-center ">
              <Image
                src="/homepage/latest4.png"
                alt="latest product"
                width={267}
                height={277}
              />
            </div>
            <div className="latest-prod-details flex justify-between items-center">
              <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
              </div>
              <p className="latest-prod-price">
                $42.00 <span className="latest-discounted">$62.00</span>
              </p>
            </div>
          </div>

          <div className="latest-prod">
            <div className="latest-prod-img flex justify-center items-center ">
              <Image
                src="/homepage/latest5.png"
                alt="latest product"
                width={303}
                height={264}
              />
            </div>
            <div className="latest-prod-details flex justify-between items-center">
              <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
              </div>
              <p className="latest-prod-price">
                $42.00 <span className="latest-discounted">$62.00</span>
              </p>
            </div>
          </div>

          <div className="latest-prod">
            <div className="latest-prod-img flex justify-center items-center ">
              <Image
                src="/homepage/latest6.png"
                alt="latest product"
                width={360}
                height={261}
              />
            </div>
            <div className="latest-prod-details flex justify-between items-center">
              <div>
                <p className="latest-prod-name">Comfort Handy Craft</p>
                <div className="underline"></div>
              </div>
              <p className="latest-prod-price">
                $42.00 <span className="latest-discounted">$62.00</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Offer Section / Section 4 */}
      <div className="offer-sec flex flex-col justify-center items-center">
        <h2>What Shopex Offer!</h2>
        <div className="offer-cards-sec grid grid-cols-4 gap-x-8">
          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/free-delivery.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/cashback.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/premium-quality.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>

          <div className="offer-card flex flex-col justify-center items-center text-center gap-y-5">
            <Image
              src="/homepage/24-hours-support.png"
              alt="free delivery"
              width={65}
              height={65}
            />
            <p className="offer-card-p">24/7 Support</p>
            <p className="offer-card-descrip">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. <br />
              Massa purus gravida.
            </p>
          </div>
        </div>
      </div>

      {/* Mid Hero Section / Section 5 */}
      <div className="mid-hero-banner flex justify-center ">
        <div className="mid-sec grid grid-cols-2 items-center justify-center">
          <div className="mid-col-1">
            <Image
              src="/homepage/mid-sec-banner.png"
              alt="Sofa Picture"
              width={558}
              height={550}
            />
          </div>
          <div className="mid-col-2 ">
            <h3>Unique Features Of latest & Trending Poducts</h3>
            <ul className="flex flex-col gap-y-4">
              <li className="flex gap-2 ">
                <div className="dot-1 "></div>All frames constructed with
                hardwood solids and laminates
              </li>
              <li className="flex gap-2 ">
                <div className="dot-2 "></div>Reinforced with double wood
                dowels, glue, screw - nails corner <br />
                blocks and machine nails
              </li>
              <li className="flex gap-2 items-center">
                <div className="dot-3 "></div>Arms, backs and seats are
                structurally reinforced
              </li>
            </ul>
            <div className="flex items-center gap-x-3 mt-5 mb-5">
              <button className="mid-col-2-btn">Shop Now</button>
              <p className="mid-col-2-p">
                B&B Italian Sofa <br />
                <span>$32.00</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Trending Section  / Section 6 */}
      <div className="trending-sec flex flex-col justify-center items-center ">
        <h2>Trending Products</h2>
        <div className="trending-inner">
          <div className="trendng-col-1 grid grid-cols-4">
            <div className="trending-prod-1 flex flex-col ">
              <div className="trending-1-img flex items-center justify-center self-center">
                <Image
                  src="/homepage/trending1.png"
                  alt="trending chair"
                  width={171}
                  height={171}
                />
              </div>
              <div className="trending-prod-details">
                <p className="p1">Canteliver Chair</p>
                <p className="p2">
                  $26.00 <span>$42.00</span>
                </p>
              </div>
            </div>
            <div className="trending-prod-1 flex flex-col ">
              <div className="trending-1-img flex items-center justify-center self-center">
                <Image
                  src="/homepage/trending2.png"
                  alt="trending chair"
                  width={171}
                  height={171}
                />
              </div>
              <div className="trending-prod-details">
                <p className="p1">Canteliver Chair</p>
                <p className="p2">
                  $26.00 <span>$42.00</span>
                </p>
              </div>
            </div>
            <div className="trending-prod-1 flex flex-col ">
              <div className="trending-1-img flex items-center justify-center self-center">
                <Image
                  src="/homepage/trending3.png"
                  alt="trending chair"
                  width={171}
                  height={171}
                />
              </div>
              <div className="trending-prod-details">
                <p className="p1">Canteliver Chair</p>
                <p className="p2">
                  $26.00 <span>$42.00</span>
                </p>
              </div>
            </div>
            <div className="trending-prod-1 flex flex-col ">
              <div className="trending-1-img flex items-center justify-center self-center">
                <Image
                  src="/homepage/trending4.png"
                  alt="trending chair"
                  width={171}
                  height={171}
                />
              </div>
              <div className="trending-prod-details">
                <p className="p1">Canteliver Chair</p>
                <p className="p2">
                  $26.00 <span>$42.00</span>
                </p>
              </div>
            </div>
          </div>
          <div className="trending-col-2 flex  gap-x-5 mt-7">
            <div className="trending-prod-2 ">
              <div className="prod-2-details">
                <h3>23% off in all products</h3>
                <p>Shop Now</p>
              </div>
              <Image
                src="/homepage/trending5.png"
                alt="trending chair"
                width={213}
                height={207}
              />
            </div>
            <div className="trending-prod-3 ">
              <div className="prod-3-details">
                <h3>23% off in all products</h3>
                <p>View Collection</p>
              </div>
              <Image
                src="/homepage/trending6.png"
                alt="trending chair"
                width={312}
                height={173}
              />
            </div>
            <div className="flex flex-col items-center justify-around ">
              <div className="trending-prod-4 flex gap-x-3 items-center">
                <div className="prod-4-img flex justify-center">
                  <Image
                    src="/homepage/trending7.png"
                    alt="trending chair"
                    width={64}
                    height={71}
                  />
                </div>
                <div className="prod-4-details">
                  <p>Executive Seat chair</p>
                  <p>
                    <span> $32.00</span>
                  </p>
                </div>
              </div>
              <div className="trending-prod-4 flex gap-x-3 items-center">
                <div className="prod-4-img flex justify-center">
                  <Image
                    src="/homepage/trending7.png"
                    alt="trending chair"
                    width={64}
                    height={71}
                  />
                </div>
                <div className="prod-4-details">
                  <p>Executive Seat chair</p>
                  <p>
                    <span> $32.00</span>
                  </p>
                </div>
              </div>
              <div className="trending-prod-4 flex gap-x-3 items-center">
                <div className="prod-4-img flex justify-center">
                  <Image
                    src="/homepage/trending7.png"
                    alt="trending chair"
                    width={64}
                    height={71}
                  />
                </div>
                <div className="prod-4-details">
                  <p>Executive Seat chair</p>
                  <p>
                    <span> $32.00</span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Discount Section / Section 7 */}
      <div className="discount-sec flex flex-col items-center justify-center">
        <div className="discount-inner ">
          <div className="discount-top text-center ">
            <h2>Discount Item</h2>
            <div className="flex justify-center gap-x-4 mt-3">
              <div className="flex items-center gap-x-2">
                <p className="except">Wood Chair </p>
                <div className="pinkdot"></div>
              </div>
              <p>Plastic Chair</p>
              <p>Sofa Collection</p>
            </div>
          </div>
          <div className="discount-bottom grid grid-cols-2 items-center ">
            <div className="discount-col-1">
              <h3>20% Discount Of All Products</h3>
              <p className="mt-4 mb-3">Eams Sofa Compact</p>
              <p className="except mb-3">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Eu eget
                feugiat habitasse nec, bibendum condimentum.
              </p>
              <div className="grid grid-cols-2 ">
                <div>
                  <p className="except flex mb-4 gap-x-2">
                    <Image
                      src="/icons/tick.svg"
                      alt="tick"
                      width={16}
                      height={11}
                    />
                    Material expose like metals
                  </p>
                  <p className="except flex mb-4 gap-x-2">
                    <Image
                      src="/icons/tick.svg"
                      alt="tick"
                      width={16}
                      height={11}
                    />
                    Simple neutral colours.
                  </p>
                </div>
                <div>
                  <p className="except flex mb-4 gap-x-2">
                    <Image
                      src="/icons/tick.svg"
                      alt="tick"
                      width={16}
                      height={11}
                    />
                    Clear lines and geomatric figures
                  </p>
                  <p className="except flex mb-4 gap-x-2">
                    <Image
                      src="/icons/tick.svg"
                      alt="tick"
                      width={16}
                      height={11}
                    />
                    Material expose like metals
                  </p>
                </div>
              </div>
              <button>Shop Now</button>
            </div>
            <div className="discount-col-2">
              <Image
                src="/homepage/discount-sofa.png"
                alt="Discount Sofa"
                width={699}
                height={597}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Top Categories Section/ Section 8 */}
      <div className="top-cate-sec flex flex-col items-center">
        <div className="top-cate-inner flex flex-col items-center">
          <h2>Top Categories</h2>
          <div className="top-cate-prod flex gap-6 mt-9">
            <div className="top-prod ">
              <div className="top-prod-img top-prod-1-img flex justify-center items-center">
                <Image
                  src="/homepage/top-prod1.png"
                  alt="Top Category Chair"
                  width={178}
                  height={178}
                />
              </div>
              <div className="top-prod-details">
                <p>Mini LCW Chair</p>
                <p>$56.00</p>
              </div>
            </div>
            <div className="top-prod ">
              <div className="top-prod-img flex justify-center items-center">
                <Image
                  src="/homepage/top-prod2.png"
                  alt="Top Category Chair"
                  width={178}
                  height={178}
                />
              </div>
              <div className="top-prod-details">
                <p>Mini LCW Chair</p>
                <p>$56.00</p>
              </div>
            </div>
            <div className="top-prod ">
              <div className="top-prod-img  flex justify-center items-center">
                <Image
                  src="/homepage/top-prod3.png"
                  alt="Top Category Chair"
                  width={178}
                  height={178}
                />
              </div>
              <div className="top-prod-details">
                <p>Mini LCW Chair</p>
                <p>$56.00</p>
              </div>
            </div>
            <div className="top-prod ">
              <div className="top-prod-img  flex justify-center items-center">
                <Image
                  src="/homepage/top-prod1.png"
                  alt="Top Category Chair"
                  width={178}
                  height={178}
                />
              </div>
              <div className="top-prod-details">
                <p>Mini LCW Chair</p>
                <p>$56.00</p>
              </div>
            </div>
          </div>
          <div className="top-cate-pagi">
            <button className="top-pagi-1"></button>
            <button className="top-pagi-2"></button>
            <button className="top-pagi-3"></button>
          </div>
        </div>
      </div>

      {/*Newsletter Banner/ Section 9  */}
      <div className="newsletter-banner flex flex-col items-center justify-center">
        <h2>
          Get Latest Update By Subscribe
          <br />
          0ur Newslater
        </h2>
        <button>Shop Now</button>
      </div>


      {/* Partnered Firms / Section 10  */}
      <div className="firms-sec flex items-center justify-center mt-24 mb-10">
        <Image src="/homepage/partnered-firms.png" alt="Partnered Firms" width={903} height={93} />
      </div>

      {/* Latest Blogs/ Section 11 */}
      <div className="latest-blog-sec flex flex-col items-center justify-center ">
        <div className="latest-blog-inner">
          <h2>Latest Blogs</h2>
          <div className="latest-blogs grid grid-cols-3">
            <div className="latest-blog-1">
              <div className="latest-blog-img">
                <Image src="/homepage/latest-blog1.jpeg" alt="Blog 1 pic" width={370} height={255}/>
              </div>
              <div className="latest-blog-details m-3">
                <div className="blog-author flex mb-5">
                  <p className="flex gap-2"> <Image src="/icons/pen.svg" alt="Pen icon" width={11.33} height={11.33} /> SaberAli</p>
                  <p className="flex gap-2"> <Image src="/icons/calender.svg" alt="Calendar icon" width={9.7} height={10.03 } /> 21 August,2020</p>
                </div>
                <h3 className="blog-title">Top esssential Trends in 2021</h3>
                <p className="blog-descrip mt-2">More off this less hello samlande lied much <br />
                over tightly circa horse taped mightly</p>
                <button className="blog-readmore-btn mt-3">Read More</button>
              </div>
            </div>
            <div className="latest-blog-1 mid-blog">
              <div className="latest-blog-img">
                <Image src="/homepage/latest-blog2.jpeg" alt="Blog 2 pic" width={370} height={255}/>
              </div>
              <div className="latest-blog-details m-3">
                <div className="blog-author flex mb-5">
                  <p className="flex gap-2"> <Image src="/icons/pen.svg" alt="Pen icon" width={11.33} height={11.33} /> SaberAli</p>
                  <p className="flex gap-2"> <Image src="/icons/calender.svg" alt="Calendar icon" width={9.7} height={10.03 } /> 21 August,2020</p>
                </div>
                <h3 className="blog-title">Top esssential Trends in 2021</h3>
                <p className="blog-descrip mt-2">More off this less hello samlande lied much <br />
                over tightly circa horse taped mightly</p>
                <button className="blog-readmore-btn mt-3">Read More</button>
              </div>
            </div>
            <div className="latest-blog-1">
              <div className="latest-blog-img">
                <Image src="/homepage/latest-blog3.jpeg" alt="Blog 3 pic" width={370} height={255}/>
              </div>
              <div className="latest-blog-details m-3">
                <div className="blog-author flex mb-5">
                  <p className="flex gap-2"> <Image src="/icons/pen.svg" alt="Pen icon" width={11.33} height={11.33} /> SaberAli</p>
                  <p className="flex gap-2"> <Image src="/icons/calender.svg" alt="Calendar icon" width={9.7} height={10.03 } /> 21 August,2020</p>
                </div>
                <h3 className="blog-title">Top esssential Trends in 2021</h3>
                <p className="blog-descrip mt-2">More off this less hello samlande lied much <br />
                over tightly circa horse taped mightly</p>
                <button className="blog-readmore-btn mt-3">Read More</button>
              </div>
            </div>
          </div>
        </div>
      </div>




    </div>
  );
}
